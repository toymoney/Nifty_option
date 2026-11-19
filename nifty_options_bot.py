#!/usr/bin/env python3
"""Nifty Options Research Bot. Read-only Upstox; Telegram; no order API.
See README.md for assumptions, deployment and limitations. Python 3.11+.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, time as dt_time
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests

IST = ZoneInfo('Asia/Kolkata')
INDEX = 'NSE_INDEX|Nifty 50'
ROOT = Path(__file__).resolve().parent
STATE_DIR = Path(os.getenv('STATE_DIR', str(ROOT / 'state')))
MIN_OI = int(os.getenv('MIN_OI', '1000'))
MIN_VOLUME = int(os.getenv('MIN_VOLUME', '100'))
MAX_SPREAD = float(os.getenv('MAX_SPREAD', '0.05'))
FAIR_GAP = float(os.getenv('FAIR_GAP', '0.15'))
COST_PCT = float(os.getenv('ROUND_TRIP_COST_PCT', '0.5'))


def now():
    return datetime.now(IST)


def parse(s):
    return datetime.fromisoformat(s).astimezone(IST)


def stamp(t):
    return t.isoformat()


def num(v, default=0.0):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def atm(spot):
    return int(math.floor(spot / 50 + 0.5) * 50)


def bs(s, k, t, vol, side, r=0.06, q=0.012):
    """European spot Black-Scholes with continuous dividend yield."""
    if min(s, k, t, vol) <= 0:
        return max(0., s-k if side == 'CE' else k-s)
    n = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2
    d1 = (math.log(s/k) + (r-q+vol*vol/2)*t) / (vol*math.sqrt(t))
    d2 = d1-vol*math.sqrt(t)
    if side == 'CE':
        return s*math.exp(-q*t)*n(d1)-k*math.exp(-r*t)*n(d2)
    return k*math.exp(-r*t)*n(-d2)-s*math.exp(-q*t)*n(-d1)


def liquid(c):
    m = c['market']
    bid, ask, ltp = (num(m.get(x)) for x in ('bid_price','ask_price','ltp'))
    return (ltp >= 10 and bid > 0 and ask >= bid and
            (ask-bid)/max((ask+bid)/2, 1) <= MAX_SPREAD and
            num(m.get('oi')) >= MIN_OI and num(m.get('volume')) >= MIN_VOLUME)


def peer_iv(c, contracts):
    """Leave-one-out IV: interpolate OI/spread-weighted IV on each side.
    Own option IV is never used to generate its reference price.
    """
    peers = [p for p in contracts if p['side'] == c['side'] and p['key'] != c['key']
             and liquid(p) and 0 < num(p['greeks'].get('iv')) < 200]
    below = sorted([p for p in peers if p['strike'] < c['strike']],
                   key=lambda p: -p['strike'])[:2]
    above = sorted([p for p in peers if p['strike'] > c['strike']],
                   key=lambda p: p['strike'])[:2]
    if not below or not above:
        return None  # no unsupported wing extrapolation
    def avg(group):
        weights = [math.sqrt(num(p['market']['oi'])) /
                   max(num(p['market']['ask_price'])-num(p['market']['bid_price']), .05)
                   for p in group]
        total = sum(weights)
        return (sum(w*p['strike'] for w,p in zip(weights,group))/total,
                sum(w*num(p['greeks']['iv']) for w,p in zip(weights,group))/total)
    kl, vl = avg(below)
    kh, vh = avg(above)
    return (vl+(vh-vl)*(c['strike']-kl)/(kh-kl))/100


def candle_hit(trade, bar):
    """Never infer first touch from OHLC if both barriers occur in a bar."""
    high, low = num(bar[2]), num(bar[3])
    if trade['strategy'] == 'S3':
        hit = high >= trade['target'] if trade['direction'] == 'UP' else low <= trade['target']
        return 'VALUE_REACHED' if hit else None
    target, stop = high >= trade['target'], low <= trade['stop']
    if target and stop:
        return 'AMBIGUOUS'
    return 'TARGET' if target else 'STOP' if stop else None


def features(c, forecast, t):
    m,g = c['market'],c['greeks']
    oi = num(m.get('oi'))
    prev = num(m.get('prev_oi'))
    return [1., min(abs(forecast)*100, 2), min(num(g.get('iv'))/50, 2),
            abs(num(g.get('delta'))), min(num(g.get('gamma'))*1000, 2),
            min(abs(num(g.get('theta')))/100, 2), min(num(g.get('vega'))/100, 2),
            max(-1,min(1,(oi-prev)/max(prev,1))),
            min((num(m.get('ask_price'))-num(m.get('bid_price')))/max(num(m.get('ltp')),1)*10,1),
            (t.hour*60+t.minute-555)/375]


def probability(weights, x):
    z = max(-20,min(20,sum(a*b for a,b in zip(weights,x))))
    return 1/(1+math.exp(-z))


class API:
    def __init__(self):
        token = os.getenv('UPSTOX_ACCESS_TOKEN','').strip()
        if not token:
            raise RuntimeError('Missing UPSTOX_ACCESS_TOKEN secret')
        self.s = requests.Session()
        self.s.headers.update({'Authorization':f'Bearer {token}', 'Accept':'application/json'})
        self.cache = {}

    def get(self, path, params=None):
        for attempt in range(4):
            try:
                r = self.s.get('https://api.upstox.com'+path, params=params, timeout=25)
            except requests.RequestException:
                if attempt == 3:
                    raise RuntimeError('Upstox network error') from None
                time.sleep(2**attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(30, max(2**attempt, num(r.headers.get('Retry-After'), 1))))
                continue
            if r.status_code != 200:
                raise RuntimeError(f'Upstox HTTP {r.status_code}; check token/data permissions')
            body = r.json()
            if body.get('status') != 'success':
                raise RuntimeError('Upstox returned unsuccessful response')
            return body['data']
        raise RuntimeError('Upstox retry limit')

    def candles(self, key, t, unit='minutes', interval=1, history=False):
        cachekey = (key, unit, interval, history, t.strftime('%Y-%m-%d %H:%M') if not history else t.date().isoformat())
        if cachekey not in self.cache:
            encoded = quote(key, safe='')
            if history:
                end = (t.date()-timedelta(days=1)).isoformat()
                start = (t.date()-timedelta(days=18)).isoformat()
                path = f'/v3/historical-candle/{encoded}/{unit}/{interval}/{end}/{start}'
            elif t.date() < now().date():
                session = t.date().isoformat()
                path = f'/v3/historical-candle/{encoded}/{unit}/{interval}/{session}/{session}'
            else:
                path = f'/v3/historical-candle/intraday/{encoded}/{unit}/{interval}'
            rows = self.get(path)['candles']
            self.cache[cachekey] = sorted(rows, key=lambda r: r[0])
            if len(self.cache) > 500:
                self.cache = {cachekey:self.cache[cachekey]}
        return self.cache[cachekey]

    def chain(self, expiry):
        rows = self.get('/v2/option/chain', {'instrument_key':INDEX,'expiry_date':expiry})
        contracts=[]
        for row in rows:
            for name, side in [('call_options','CE'),('put_options','PE')]:
                obj=row.get(name) or {}
                if obj.get('instrument_key'):
                    contracts.append(dict(key=obj['instrument_key'], strike=num(row['strike_price']),
                        side=side, expiry=row['expiry'], market=obj.get('market_data') or {},
                        greeks=obj.get('option_greeks') or {}, spot=num(row['underlying_spot_price'])))
        return contracts


class DailyJournal:
    """Auditable daily lessons. Does not alter Kronos or S2/S3 entry rules.
    Policy is frozen after each review and can only affect later dates.
    """
    LABELS = {
        'side:CE':'buying calls', 'side:PE':'buying puts',
        'iv:normal':'options with IV below 25%', 'iv:high':'options with IV of 25% or more',
        'spread:tight':'options with a bid/ask gap of 1.5% or less',
        'spread:wide':'options with a bid/ask gap above 1.5%',
        'forecast:strong':'Kronos moves of at least 0.20%',
        'forecast:small':'Kronos moves below 0.20%',
        'time:morning':'entries before noon', 'time:afternoon':'entries after noon',
        'oi:rising':'open interest rising by at least 10%',
        'oi:falling':'open interest falling by at least 10%', 'oi:steady':'mostly steady open interest',
    }

    def __init__(self,state):
        self.state=state
        self.data=state.setdefault('journal',{'version':1,'reviews':{},'rules':{},'events':[]})

    @staticmethod
    def tags(c,forecast,t):
        m,g=c.get('market',{}),c.get('greeks',{})
        tags=['side:'+c['side'],'time:'+('morning' if t.hour<12 else 'afternoon')]
        if g.get('iv') is not None:
            tags.append('iv:'+('high' if num(g['iv'])>=25 else 'normal'))
        if num(m.get('ltp'))>0 and m.get('ask_price') is not None and m.get('bid_price') is not None:
            spread=(num(m['ask_price'])-num(m['bid_price']))/num(m['ltp'])
            tags.append('spread:'+('tight' if spread<=.015 else 'wide'))
        if forecast is not None:
            tags.append('forecast:'+('strong' if abs(forecast)>=.002 else 'small'))
        if num(m.get('prev_oi'))>0:
            change=(num(m.get('oi'))-num(m['prev_oi']))/num(m['prev_oi'])
            tags.append('oi:'+('rising' if change>=.10 else 'falling' if change<=-.10 else 'steady'))
        return tags

    def trade_tags(self,tr):
        if 'journal_tags' in tr:
            return tr['journal_tags']
        if tr.get('features_snapshot') and tr.get('entered'):
            return self.tags(tr['features_snapshot'],tr.get('forecast_return'),parse(tr['entered']))
        return []

    def sessions(self,date):
        dates={d for d,day in self.state['days'].items() if d<=date and day.get('market_seen')}
        dates.update(z['day'] for z in self.state['trades'] if z['day']<=date)
        return sorted(dates)

    def samples(self,date):
        dates=set(self.sessions(date)[-30:])
        return [z for z in self.state['trades'] if z['day'] in dates and z['strategy']=='S1'
                and z['status'] in ('TARGET','STOP') and z.get('net_estimated_pct') is not None
                and (not z.get('shadow') or z.get('journal_role')=='skipped_candidate')
                and (not z.get('closed') or parse(z['closed']).date().isoformat()<=date)]

    @staticmethod
    def stats(rows):
        return {'cases':len(rows),'days':len({z['day'] for z in rows}),
                'wins':sum(z['status']=='TARGET' for z in rows),
                'losses':sum(z['status']=='STOP' for z in rows),
                'mean_net_pct':statistics.mean(z['net_estimated_pct'] for z in rows) if rows else None}

    def policies(self,date):
        # Never use this evening's results for this morning's decisions.
        dates=[d for d in self.data['reviews'] if d<date]
        if not dates:return []
        review=self.data['reviews'][max(dates)]
        return review.get('tomorrow_policies', [review['tomorrow_policy']] if review.get('tomorrow_policy') else [])

    def policy(self,date):
        policies=self.policies(date)
        return policies[0] if policies else None

    def decide(self,date,tags,score,trained):
        base_allowed=trained<30 or score>=.45
        result={'allow':base_allowed,'base_allowed':base_allowed,'changed':False,
                'rule_id':None,'reason':'usual learned-score filter' if not base_allowed else 'usual setup rules'}
        matches=[r for r in self.policies(date) if r['tag'] in tags]
        avoid=[r for r in matches if r['action']=='avoid']
        # Avoid has priority; attribute the effect to ONE rule, never double-credit.
        if avoid:
            if base_allowed:
                rule=avoid[0]
                result.update(allow=False,changed=True,rule_id=rule['id'],
                              reason='daily journal: avoid '+self.LABELS[rule['tag']])
            return result
        prefer=[r for r in matches if r['action']=='prefer']
        if prefer and not base_allowed and trained>=30 and score>=.40:
            rule=prefer[0]
            result.update(allow=True,changed=True,rule_id=rule['id'],
                          reason='daily journal: small trial favouring '+self.LABELS[rule['tag']])
        return result

    def review(self,day):
        date=day['date']
        if date in self.data['reviews']:
            return self.data['reviews'][date]
        sessions=self.sessions(date)
        sample=self.samples(date)
        prior=[d for d in self.data['reviews'] if d<date]
        previous=self.data['reviews'][max(prior)] if prior else None
        active_rules=[]
        if previous:
            policies=previous.get('tomorrow_policies', [previous['tomorrow_policy']] if previous.get('tomorrow_policy') else [])
            active_rules=[self.data['rules'][r['id']] for r in policies if r['id'] in self.data['rules']]
        actions=[]
        evaluated_rules=[]
        for active in active_rules:
            affected=[z for z in sample if z.get('journal_rule_id')==active['id'] and z.get('journal_changed')
                      and z['day']>active['created']]
            # Actual new alerts for prefer; counterfactual skipped candidates for avoid.
            effects=[(-1 if active['action']=='avoid' else 1)*z['net_estimated_pct'] for z in affected]
            active['forward']={**self.stats(affected),
                'mean_effect_pct':statistics.mean(effects) if effects else None,
                'helped':sum(x>0 for x in effects),'hurt':sum(x<0 for x in effects)}
            age=len([d for d in sessions if d>active['created']])
            evidence=len(effects)>=10 and len({z['day'] for z in affected})>=5
            status=active['status']
            if evidence and statistics.mean(effects)>2:
                active['status']='KEEP'
                note='The trial helped on later cases; keep it and continue checking.'
            elif evidence and statistics.mean(effects)<-2:
                active['status']='PAUSED'
                note='The trial hurt on later cases; remove it from tomorrow\'s decisions.'
            elif age>=10:
                active['status']='PAUSED'
                note='The trial has not shown enough benefit after 10 sessions; stop using it.'
            else:
                note='The trial still needs more later outcomes; keep testing one change.'
            if status!=active['status']:
                self.data['events'].append({'date':date,'rule_id':active['id'],
                    'from':status,'to':active['status'],'reason':note})
                if active['status']=='PAUSED':active['paused_on']=date
            actions.append(note)
            evaluated={'id':active['id'],'tag':active['tag'],'action':active['action'],
                       'status':active['status'],'forward':dict(active['forward'])}
            evaluated_rules.append(evaluated)
        active_rules=[r for r in active_rules if r['status']!='PAUSED']

        # Predefined, broad tags limit post-hoc rule searching. Descriptive, not causal.
        patterns=[]
        for tag,label in self.LABELS.items():
            group=[z for z in sample if tag in self.trade_tags(z)]
            other=[z for z in sample if tag not in self.trade_tags(z)
                   and any(k.split(':')[0]==tag.split(':')[0] for k in self.trade_tags(z))]
            if not group:continue
            stat=self.stats(group); comparison=self.stats(other)
            enough=stat['cases']>=12 and stat['days']>=5 and comparison['cases']>=6 and comparison['days']>=3
            difference=stat['mean_net_pct']-comparison['mean_net_pct'] if other else None
            action=None
            if enough and stat['wins']/stat['cases']<=.35 and stat['mean_net_pct']<0 and difference<=-4:
                action='avoid'
            if enough and stat['wins']/stat['cases']>=.65 and stat['mean_net_pct']>0 and difference>=4:
                action='prefer'
            patterns.append({'tag':tag,'label':label,'stats':stat,'comparison':comparison,
                             'difference_pct':difference,'candidate_action':action})
        candidates=[p for p in patterns if p['candidate_action']]
        candidates.sort(key=lambda p:abs(p['difference_pct']),reverse=True)
        if len(active_rules)<3 and not any(r['status']=='TRY' for r in active_rules):
            for p in candidates:
                # One rule per feature family; never stack contradictory variants.
                if any(r['tag'].split(':')[0]==p['tag'].split(':')[0] for r in active_rules):
                    continue
                # Do not retry a failed lesson immediately; preserve its failed history.
                same=[r for r in self.data['rules'].values() if r['tag']==p['tag'] and r['action']==p['candidate_action']]
                if any(r.get('paused_on') and len([d for d in sessions if d>r['paused_on']])<5 for r in same):
                    continue
                if p['candidate_action']=='prefer' and len(self.state['learner']['trained_ids'])<30:
                    continue
                rid=hashlib.sha256(f'{date}:{p["tag"]}:{p["candidate_action"]}'.encode()).hexdigest()[:12]
                active={'id':rid,'tag':p['tag'],'action':p['candidate_action'],'status':'TRY',
                        'created':date,'discovery':p,'forward':{'cases':0,'days':0,'mean_effect_pct':None}}
                self.data['rules'][rid]=active
                active_rules.append(active)
                self.data['events'].append({'date':date,'rule_id':rid,'from':'OBSERVE','to':'TRY',
                                           'reason':'Enough past examples for one limited forward trial.'})
                actions.append('Start one small trial tomorrow: '+('avoid ' if active['action']=='avoid' else 'favour ')+p['label']+'.')
                break
        frozen_rules=[{k:r[k] for k in ('id','tag','action','status','created')} for r in active_rules]
        frozen=frozen_rules[0] if frozen_rules else None
        active=active_rules[0] if active_rules else None
        evaluated=evaluated_rules[0] if evaluated_rules else None
        today=[z for z in self.state['trades'] if z['day']==date and not z.get('shadow')]
        review={'date':date,'session_number':len(sessions),'market_session':date in sessions,
                'today_summary':{s:{'alerts':len([z for z in today if z['strategy']==s]),
                   'target':len([z for z in today if z['strategy']==s and z['status'] in ('TARGET','VALUE_REACHED')]),
                   'stop':len([z for z in today if z['strategy']==s and z['status']=='STOP']),
                   'unresolved':len([z for z in today if z['strategy']==s and z['status'] not in ('TARGET','STOP','VALUE_REACHED')])}
                    for s in ('S1','S2','S3')},
                'patterns':patterns,'actions':actions,'tomorrow_policy':frozen,'evaluated_rule':evaluated,
                'tomorrow_policies':frozen_rules,'evaluated_rules':evaluated_rules,
                'active_rules':[{**r,'forward':dict(r['forward'])} for r in active_rules],
                'active_forward':dict(active.get('forward',{})) if active else None,
                'new_training_cases':day.get('new_training_cases',0),
                'milestone':date in sessions and len(sessions)>0 and len(sessions)%15==0}
        if review['milestone']:
            review['checkpoint_results']={}
            for strategy in ('S1','S2','S3'):
                windows=[]
                for label,window in [('latest 15 sessions',sessions[-15:]),('previous 15 sessions',sessions[-30:-15])]:
                    rows=[z for z in self.state['trades'] if z['day'] in window and z['strategy']==strategy and not z.get('shadow')]
                    windows.append({'window':label,'sessions':len(window),'alerts':len(rows),
                        'hits':sum(z['status'] in ('TARGET','VALUE_REACHED') for z in rows),
                        'stops':sum(z['status']=='STOP' for z in rows),
                        'unresolved':sum(z['status'] not in ('TARGET','STOP','VALUE_REACHED') for z in rows)})
                review['checkpoint_results'][strategy]=windows
        self.data['reviews'][date]=review
        return review

    def english(self,review):
        date=review['date']
        today=[z for z in self.state['trades'] if z['day']==date and not z.get('shadow')]
        won=[z for z in today if z['status'] in ('TARGET','VALUE_REACHED')]
        lost=[z for z in today if z['status']=='STOP']
        unclear=[z for z in today if z['status'] in ('AMBIGUOUS','DATA_MISSING','OPEN','EOD_OPEN')]
        lines=[f'MY DAILY LEARNING REVIEW | {date}',f'Recorded market session: {review["session_number"]}',
               '\nWHAT I DID WELL']
        if won:
            for z in won[:4]:
                lines.append(f'- {z["strategy"]}: {z["strike"]:g} {z["side"]} reached '+('its model value.' if z['strategy']=='S3' else 'the target.'))
        else:lines.append('- I have no verified target/value hit to call a success today.')
        lines.append('\nWHAT WENT WRONG')
        if lost:
            for z in lost[:4]:
                lines.append(f'- {z["strategy"]}: {z["strike"]:g} {z["side"]} hit its stop. I saved the entry conditions to compare with future results.')
        else:lines.append('- No verified stop hit today. This alone does not prove the decisions were good.')
        if unclear:lines.append(f'- {len(unclear)} alert(s) remain unresolved or uncertain; I will not call them wins.')
        lines.append('\nWHAT THE RECORD IS TEACHING ME')
        patterns=review['patterns']
        positive=sorted([p for p in patterns if p['stats']['mean_net_pct']>0],key=lambda p:-p['stats']['mean_net_pct'])
        negative=sorted([p for p in patterns if p['stats']['mean_net_pct']<0],key=lambda p:p['stats']['mean_net_pct'])
        for label,items in [('Encouraging',positive),('Weak',negative)]:
            if items:
                p=items[0];s=p['stats']
                lines.append(f'- {label}: {p["label"]}: {s["wins"]} targets and {s["losses"]} stops across {s["days"]} sessions. '
                             f'Average paper result after assumed costs: {s["mean_net_pct"]:+.1f}% of entry premium. '
                             +('Enough past examples for a trial, not proof.' if p['candidate_action'] else 'An observation only; not enough comparative evidence for a new rule.'))
        if not patterns:lines.append('- No resolved S1 examples yet. I will record the next session rather than invent a lesson.')
        lines.append(f'- I updated my small scoring model from {review["new_training_cases"]} newly resolved alert(s). Kronos was not changed.')
        lines.append('\nWHAT I WILL TRY TOMORROW')
        policies=review.get('active_rules',[])
        for policy in policies:
            label=self.LABELS[policy['tag']]
            if policy['action']=='avoid':
                lines.append(f'- {policy["status"]}: skip S1 alerts for {label}. I will still track these silently, including missed winners.')
            else:
                lines.append(f'- {policy["status"]}: give {label} a small preference: allow S1 scores from 0.40 instead of 0.45. All other entry checks stay in place.')
            f=policy.get('forward') or {}
            if f.get('cases'):
                lines.append(f'- On {f["cases"]} later affected case(s), this change helped {f.get("helped",0)} and hurt {f.get("hurt",0)}. '
                             f'Average measured effect: {f["mean_effect_pct"]:+.1f} percentage points of entry premium per affected case, not portfolio profit.')
        if not policies:
            lines.append('- Keep the entry rules while I collect more evidence. No extra filter is justified yet.')
        for evaluated in review.get('evaluated_rules',[]):
            if evaluated['status']=='PAUSED':
                f=evaluated['forward']
                lines.append(f'- Removed trial: {evaluated["action"]} {self.LABELS[evaluated["tag"]]}. Later affected cases: {f["cases"]}; helped {f.get("helped",0)}, hurt {f.get("hurt",0)}.')
        lines.extend('- '+a for a in review['actions'])
        lines.append('- I will keep reviewing the opening and value-watch results. Their original rules stay fixed.')
        lines.append('\nMY 15–30 SESSION RECORD')
        records=[r for r in self.data['rules'].values() if r['created']<=date]
        lines.append(f'- {len(records)} trial(s) recorded. Every keep/pause decision and daily plan is saved.')
        if review['milestone']:
            lines.append(f'- Session {review["session_number"]} review reached. Look at the saved summary before treating any repeated pattern as dependable.')
            for strategy,windows in review.get('checkpoint_results',{}).items():
                for window in windows:
                    if window['sessions']:
                        lines.append(f'- {strategy}, {window["window"]}: {window["alerts"]} alerts; {window["hits"]} target/value hits, {window["stops"]} stops, {window["unresolved"]} unresolved.')
            lines.append('- These periods can have different market conditions; a better result alone does not prove the new rules caused it.')
            for r in records[-5:]:
                lines.append(f'- {r["status"]}: {r["action"]} {self.LABELS[r["tag"]]}.')
        else:
            next_review=(review['session_number']//15+1)*15
            lines.append(f'- Next checkpoint: session {next_review}. Calendar days without observed market data do not count.')
        lines.append('These are observed patterns, not proven reasons. No forced new rule after a single win or loss.')
        return lines


class Bot:
    def __init__(self, api, directory=STATE_DIR):
        self.api=api
        self.dir=Path(directory)
        self.dir.mkdir(parents=True,exist_ok=True)
        self.path=self.dir/'state.json'
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {
            'version':1,'days':{},'trades':[], 'outbox':[],
            'learner':{'weights':[0.]*10,'trained_ids':[],'brier':[]}}
        if self.state.get('version') != 1:
            raise RuntimeError('Unsupported state version')
        self.journal=DailyJournal(self.state)
        self.predictor=None
        self.last_sync=0.
        self.last_kronos=None
        self.executor=ThreadPoolExecutor(max_workers=1)
        self.pending_forecast=None
        self.forecast_started=None

    def save(self):
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state,indent=2,allow_nan=False))
        temp.replace(self.path)

    def sync(self, force=False):
        self.save()
        if os.getenv('GIT_STATE') != 'true' or (not force and time.monotonic()-self.last_sync < 300):
            return
        def git(*args):
            p=subprocess.run(['git','-C',str(self.dir),*args],capture_output=True,text=True)
            if p.returncode:
                raise RuntimeError('State git operation failed; check Actions write permission')
            return p.stdout
        git('add','state.json','reports')
        if git('status','--porcelain').strip():
            git('commit','-m','Save research state [skip ci]')
        git('push','origin','HEAD:bot-state')
        self.last_sync=time.monotonic()

    def emit(self, event_id, message):
        if not any(x['id']==event_id for x in self.state['outbox']):
            self.state['outbox'].append({'id':event_id,'text':message,'sent':False})
            self.save()

    def flush(self):
        token=os.getenv('TELEGRAM_BOT_TOKEN','')
        chat=os.getenv('TELEGRAM_CHAT_ID','')
        if not token or not chat:
            raise RuntimeError('Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets')
        for item in self.state['outbox']:
            if item['sent']:
                continue
            try:
                r=requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                    json={'chat_id':chat,'text':item['text'], 'disable_web_page_preview':True},timeout=20)
                body=r.json()
                if not r.ok or not body.get('ok'):
                    raise RuntimeError(f'Telegram rejected message (HTTP {r.status_code})')
            except (requests.RequestException, ValueError):
                # Do not print exception URLs: Telegram embeds the secret in its URL.
                raise RuntimeError('Telegram network/response error; message retained for retry') from None
            item['sent']=True
            self.save()

    def diagnostic(self, day, key, message):
        day.setdefault('diagnostics',{})[key]=message

    def prepare(self,t):
        date=t.date().isoformat()
        day=self.state['days'].setdefault(date,{'date':date,'diagnostics':{}})
        if 'expiry' in day:
            return day
        hist=self.api.candles(INDEX,t,unit='days',history=True)
        if not hist:
            raise RuntimeError('No previous Nifty session history')
        previous=hist[-1]
        contracts=self.api.get('/v2/option/contract',{'instrument_key':INDEX})
        expiries=sorted({c['expiry'] for c in contracts if c['expiry']>=date})
        if not expiries:
            raise RuntimeError('No active Nifty expiry')
        day.update(expiry=expiries[0],previous_date=parse(previous[0]).date().isoformat(),
                   previous_spot=num(previous[4]), previous_atm=atm(num(previous[4])))
        day['opening_contracts']={}
        for c in contracts:
            if c['expiry']==day['expiry'] and num(c['strike_price'])==day['previous_atm']:
                if c['instrument_type'] in ('CE','PE'):
                    day['opening_contracts'][c['instrument_type']]=c['instrument_key']
        self.emit('online-'+date, f'Nifty research bot ONLINE | {date}\nExpiry {day["expiry"]}; previous-close ATM {day["previous_atm"]}.\nS1 Kronos + learning; S2 opening gap; S3 relative fair value.\nPaper alerts only. Daily report around 16:15 IST.')
        self.save()
        return day

    def fresh(self,key,t):
        bars=self.api.candles(key,t)
        valid=[b for b in bars if parse(b[0]).date()==t.date() and parse(b[0])+timedelta(minutes=1)<=t]
        return bool(valid and (t-parse(valid[-1][0])).total_seconds()<=180)

    def create_trade(self,strategy,c,t,day,extra=None,shadow=False):
        # S1 at most one per 30-min slot; S2/S3 once per contract per session.
        slot=t.strftime('%H')+str(t.minute//30) if strategy=='S1' else ''
        ident=hashlib.sha256(f'{day["date"]}:{strategy}:{c["key"]}:{slot}:{shadow}'.encode()).hexdigest()[:14]
        if any(z['id']==ident for z in self.state['trades']):
            return None
        entry=num(c['market'].get('ltp'))
        if entry<=0:
            return None
        trade=dict(id=ident,day=day['date'],strategy=strategy,key=c['key'],strike=c['strike'],
                   side=c['side'],expiry=c['expiry'],entry=entry,entered=stamp(t),
                   target=round(entry*1.15,2),stop=round(entry*.90,2),direction='UP',
                   status='OPEN',shadow=shadow,features_snapshot=c,
                   last=entry,last_seen=stamp(t),checked_until=stamp(t),
                   entry_minute_excluded=True)
        if extra:
            trade.update(extra)
        self.state['trades'].append(trade)
        if not shadow:
            details=(f'Fixed model value ₹{trade["target"]:.2f}; direction {trade["direction"]}\n'
                     f'Gap {trade["gap_pct"]:.1f}%; peer IV {trade["peer_iv"]:.2f}%\nNo stop: valuation watch only.'
                     if strategy=='S3' else f'Target ₹{trade["target"]:.2f} (+15%) | SL ₹{trade["stop"]:.2f} (-10%)\nNo trailing SL.')
            if strategy=='S1':
                details+=(f'\nKronos 15m forecast {trade.get("forecast_return",0)*100:+.3f}%'
                          f' | learned score {trade.get("score_at_entry",.5):.3f} (not calibrated win probability)')
            if strategy=='S3':
                g=c['greeks']; m=c['market']
                details+=(f'\nIV {num(g.get("iv")):.2f}% | Δ {num(g.get("delta")):.3f}'
                          f' | Γ {num(g.get("gamma")):.5f}\nΘ {num(g.get("theta")):.2f}'
                          f' | Vega {num(g.get("vega")):.2f} | OI {num(m.get("oi")):g}'
                          f'\nExpiry in {trade.get("time_to_expiry_years",0)*365:.2f} days')
            opening=(f'Previous close ₹{trade["previous_close"]:.2f}; open ₹{trade["opening"]:.2f}\n'
                     if strategy=='S2' else '')
            self.emit('entry-'+ident,f'{strategy} PAPER RESEARCH | NIFTY {c["strike"]:g} {c["side"]}\n'
                f'Expiry {c["expiry"]} | {t:%H:%M:%S} IST\nLTP/reference ₹{entry:.2f}\n'+opening+details+
                f'\nID {ident}')
        self.save()
        return trade

    def opening_strategy(self,day,chain,t):
        if day.get('opening_done') or t.time()<dt_time(9,16):
            return
        if t.time()>dt_time(9,25):
            day['opening_done']=True
            self.diagnostic(day,'S2','Opening alert skipped: runner/data started after 09:25 IST; no backdated entry.')
            return
        bykey={c['key']:c for c in chain}
        done=day.setdefault('opening_checked',[])
        for side,key in day['opening_contracts'].items():
            if side in done or key not in bykey:
                continue
            hist=self.api.candles(key,t,unit='days',history=True)
            prior=[b for b in hist if parse(b[0]).date().isoformat()==day['previous_date']]
            bars=self.api.candles(key,t)
            opening=[b for b in bars if parse(b[0]).date()==t.date() and parse(b[0]).time()==dt_time(9,15)]
            if not prior or not opening:
                self.diagnostic(day,'S2_'+side,'Waiting for same-contract previous close and 09:15 candle')
                continue
            c=bykey[key]
            close,op=num(prior[-1][4]),num(opening[0][1])
            day.setdefault('opening_reference',{})[side]={'close':close,'open':op,'key':key}
            if op>close>0:
                if not liquid(c) or not self.fresh(key,t):
                    continue
                self.create_trade('S2',c,t,day,{'previous_close':close,'opening':op})
            done.append(side)
        day['opening_done']=len(done)==2

    def valuation_strategy(self,day,chain,t):
        if not dt_time(9,20)<=t.time()<=dt_time(15,0):
            return
        spot=chain[0]['spot']
        center=atm(spot)
        band=[c for c in chain if abs(c['strike']-center)<=250]
        expiry=datetime.fromisoformat(day['expiry']).replace(hour=15,minute=30,tzinfo=IST)
        years=(expiry-t).total_seconds()/(365*86400)
        if (expiry-t).total_seconds()<1800:
            return
        for c in band:
            if any(z['day']==day['date'] and z['strategy']=='S3' and z['key']==c['key'] for z in self.state['trades']):
                continue
            if not liquid(c):
                continue
            vol=peer_iv(c,band)
            if vol is None:
                continue
            fair=bs(spot,c['strike'],years,vol,c['side'],
                    float(os.getenv('RISK_FREE_RATE','.06')),float(os.getenv('DIVIDEND_YIELD','.012')))
            m=c['market']; ltp=num(m.get('ltp'))
            if fair<10:
                continue
            gap=(fair-ltp)/fair
            # Gap must survive executable side of spread, not just stale LTP.
            executable=num(m['ask_price']) if gap>0 else num(m['bid_price'])
            if abs(gap)<FAIR_GAP or abs((fair-executable)/fair)<FAIR_GAP:
                continue
            if not self.fresh(c['key'],t):
                continue
            self.create_trade('S3',c,t,day,dict(target=round(fair,2),stop=None,
                direction='UP' if gap>0 else 'DOWN',gap_pct=gap*100,peer_iv=vol*100,
                time_to_expiry_years=years,
                greek_scenarios={
                    'spot_up_1pct_approx':num(c['greeks'].get('delta'))*spot*.01 + .5*num(c['greeks'].get('gamma'))*(spot*.01)**2,
                    'spot_down_1pct_approx':-num(c['greeks'].get('delta'))*spot*.01 + .5*num(c['greeks'].get('gamma'))*(spot*.01)**2,
                    'iv_up_1_percentage_point':num(c['greeks'].get('vega')),
                    'provider_theta':num(c['greeks'].get('theta'))},
                fair_model='leave_one_out_peer_IV_Black_Scholes',
                severity='25%+' if abs(gap)>=.25 else '15–25%'))

    def forecast(self,t):
        if os.getenv('ENABLE_KRONOS','true').lower()!='true':
            raise RuntimeError('Kronos disabled: S1 paused; S2/S3 remain active')
        import pandas as pd
        import torch
        sys.path.insert(0,os.getenv('KRONOS_PATH',str(ROOT/'vendor'/'Kronos')))
        from model import Kronos, KronosTokenizer, KronosPredictor
        torch.set_num_threads(2)
        torch.manual_seed(42)
        if self.predictor is None:
            tokenizer=KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-2k')
            model=Kronos.from_pretrained('NeoQuasar/Kronos-mini')
            self.predictor=KronosPredictor(model,tokenizer,device='cpu',max_context=256)
        forecast_api=API()  # isolated HTTP session for background inference
        history=forecast_api.candles(INDEX,t,interval=5,history=True)
        intraday=forecast_api.candles(INDEX,t,interval=5)
        rows={b[0]:b for b in history+intraday if parse(b[0])+timedelta(minutes=5)<=t}
        bars=[rows[k] for k in sorted(rows)][-256:]
        if len(bars)<120:
            raise RuntimeError('Kronos needs at least 120 completed five-minute candles')
        df=pd.DataFrame([b[:6] for b in bars],columns=['timestamp','open','high','low','close','volume'])
        # Nifty spot index has no traded volume. Preserve zero, never fabricate it.
        x=pd.Series([parse(b[0]).replace(tzinfo=None) for b in bars])
        y=pd.Series([x.iloc[-1]+timedelta(minutes=5*i) for i in (1,2,3)])
        pred=self.predictor.predict(df=df[['open','high','low','close','volume']],x_timestamp=x,
             y_timestamp=y,pred_len=3,T=1.,top_p=.9,sample_count=3,verbose=False)
        value=float(pred['close'].iloc[-1])/float(df['close'].iloc[-1])-1
        if not math.isfinite(value) or abs(value)>.05:
            raise RuntimeError('Kronos invalid/outlier forecast')
        return {'return_15m':value,'reference_close':float(df['close'].iloc[-1]),
                'reference_bar':bars[-1][0],
                'target_bar':stamp(parse(bars[-1][0])+timedelta(minutes=15)),
                'predicted_close':float(pred['close'].iloc[-1])}

    def learning_strategy(self,day,chain,t):
        if not dt_time(9,35)<=t.time()<=dt_time(14,45):
            return
        slot=t.strftime('%Y%m%d%H')+str(t.minute//30)
        existing=[z for z in self.state['trades'] if z['day']==day['date'] and z['strategy']=='S1' and not z['shadow']]
        if len(existing)>=3 or any(z['status']=='OPEN' for z in existing):
            return
        if self.pending_forecast is None:
            if day.get('forecast_slot')==slot:
                return
            day['forecast_slot']=slot
            self.forecast_started=t
            self.pending_forecast=self.executor.submit(self.forecast,t)
            return
        if not self.pending_forecast.done():
            return
        future=self.pending_forecast
        self.pending_forecast=None
        try:
            forecast_result=future.result()
            f=forecast_result['return_15m']
        except Exception as exc:
            self.diagnostic(day,'Kronos',f'Unavailable ({type(exc).__name__}); S1 skipped. No substitute prediction.')
            self.emit('kronos-issue-'+day['date'], 'Kronos prediction unavailable; S1 paused for this slot. S2/S3 continue. Retrying next half-hour slot; details in daily diagnostics.')
            return
        if (t-self.forecast_started).total_seconds()>180:
            self.diagnostic(day,'Kronos','Inference older than 3 minutes discarded; monitoring continued in background.')
            return
        day['diagnostics']['Kronos']='Native pretrained Kronos-mini forecast successful; no base-model fine-tuning.'
        day.setdefault('forecasts',[]).append({'at':stamp(t),**forecast_result})
        if abs(f)<.001:
            return
        side='CE' if f>0 else 'PE'
        center=atm(chain[0]['spot'])
        options=[c for c in chain if c['side']==side and c['strike']==center and liquid(c)]
        if not options:
            return
        c=options[0]
        if not self.fresh(c['key'],t):
            return
        x=features(c,f,t)
        learner=self.state['learner']; p=probability(learner['weights'],x)
        trained=len(learner['trained_ids'])
        tags=self.journal.tags(c,f,t)
        decision=self.journal.decide(day['date'],tags,p,trained)
        bucket=f'{side}:{"high_iv" if num(c["greeks"].get("iv"))>=25 else "normal_iv"}'
        metadata={'x':x,'score_at_entry':p,'trained_at_entry':trained,
            'forecast_return':f,'bucket':bucket,'model':'Kronos-mini pretrained + online logistic ranker',
            'journal_tags':tags,'journal_rule_id':decision['rule_id'],
            'journal_changed':decision['changed'],'journal_reason':decision['reason'],
            'base_allowed':decision['base_allowed']}
        if not decision['allow']:
            day.setdefault('learning_skips',[]).append({'at':stamp(t),'reasons':[decision['reason']],'score':p})
            # Follow skipped setups with the same barriers to count BOTH avoided losses
            # and missed winners; never train the live ranker on these counterfactuals.
            self.create_trade('S1',c,t,day,{**metadata,'journal_role':'skipped_candidate'},shadow=True)
            return
        main=self.create_trade('S1',c,t,day,{**metadata,'journal_role':'main'})
        if main:
            opposite=[z for z in chain if z['strike']==center and z['side']!=side and liquid(z)]
            if opposite and self.fresh(opposite[0]['key'],t):
                self.create_trade('S1',opposite[0],t,day,{'parent_id':main['id'],'journal_role':'counter',
                    'counter_definition':'Buy opposite CE/PE at same strike and time; shadow only'},shadow=True)

    def finish(self,tr,status,t,exit_price=None,evidence='completed_1m_candle'):
        tr.update(status=status,closed=stamp(t),exit_price=exit_price,evidence=evidence)
        if tr['strategy']!='S3' and exit_price is not None:
            tr['gross_pct']=(exit_price/tr['entry']-1)*100
            tr['net_estimated_pct']=tr['gross_pct']-COST_PCT
        if not tr['shadow']:
            label=('Both target and SL inside one candle: first touch unknown; excluded from training.'
                   if status=='AMBIGUOUS' else f'{status}: '+(f'₹{exit_price:.2f}' if exit_price is not None else 'no verified exit'))
            if status not in ('EOD_OPEN','DATA_MISSING'):
                self.emit('exit-'+tr['id'],f'{tr["strategy"]} | NIFTY {tr["strike"]:g} {tr["side"]}\n'
                    f'{label}\nEvidence: {evidence}\nID {tr["id"]} | {t:%H:%M} IST')

    def monitor(self,day,chain,t,closing=False):
        mapping={c['key']:c for c in chain}
        for tr in self.state['trades']:
            if tr['day']!=day['date'] or tr['status']!='OPEN':
                continue
            try:
                bars=self.api.candles(tr['key'],t)
            except RuntimeError:
                self.diagnostic(day,'monitor_'+tr['id'],'Option candles unavailable; outcome unverified')
                if closing:
                    self.finish(tr,'DATA_MISSING',t,None,'history unavailable')
                continue
            entered=parse(tr['entered'])
            # Exclude the entry minute: its high/low may predate the alert.
            start=entered.replace(second=0,microsecond=0)+timedelta(minutes=1)
            cursor=parse(tr['checked_until'])
            for b in bars:
                bt=parse(b[0])
                if bt.date()!=t.date() or bt<start or bt<=cursor or bt+timedelta(minutes=1)>t:
                    continue
                hit=candle_hit(tr,b)
                tr['checked_until']=stamp(bt)
                tr['last']=num(b[4]);tr['last_seen']=stamp(bt+timedelta(minutes=1))
                if hit:
                    price=tr['target'] if hit in ('TARGET','VALUE_REACHED') else tr['stop'] if hit=='STOP' else None
                    # A gap through stop is marked at bar open, not a fictional stop fill.
                    if hit=='STOP':
                        price=min(price,num(b[1]))
                    self.finish(tr,hit,bt+timedelta(minutes=1),price)
                    break
            if tr['status']!='OPEN':
                continue
            c=mapping.get(tr['key'])
            if not closing and c and self.fresh(tr['key'],t):
                ltp=num(c['market'].get('ltp'))
                if ltp>0:
                    tr['last']=ltp;tr['last_seen']=stamp(t)
                    # Only single-barrier watches can close on a snapshot.
                    # Two-barrier trades wait for completed candles.
                    if tr['strategy']=='S3':
                        hit=candle_hit(tr,[None,ltp,ltp,ltp,ltp])
                        if hit:
                            self.finish(tr,hit,t,ltp,'polled_LTP valuation touch')
            if closing and tr['status']=='OPEN':
                self.finish(tr,'EOD_OPEN' if parse(tr['last_seen']).time()>=dt_time(15,25) else 'DATA_MISSING',t,
                            tr['last'], 'last_available_price; neither barrier verified')

    def train(self,day):
        learner=self.state['learner']
        for tr in self.state['trades']:
            if (tr['day']>day['date'] or tr['strategy']!='S1' or tr['shadow'] or
                tr['status'] not in ('TARGET','STOP') or 'x' not in tr or tr['id'] in learner['trained_ids']):
                continue
            y=1. if tr['status']=='TARGET' else 0.
            # Frozen entry score is the forward test; no fitted-on-itself accuracy.
            learner['brier'].append((tr['score_at_entry']-y)**2)
            p=probability(learner['weights'],tr['x'])
            learner['weights']=[w+.03*((y-p)*xi-.001*w) for w,xi in zip(learner['weights'],tr['x'])]
            learner['trained_ids'].append(tr['id'])

    def report(self,day,t):
        if day.get('reported'):
            return
        before_training=len(self.state['learner']['trained_ids'])
        self.train(day)
        day['new_training_cases']=len(self.state['learner']['trained_ids'])-before_training
        if day.get('forecasts'):
            try:
                bars={stamp(parse(b[0])):b for b in self.api.candles(INDEX,t,interval=5)}
                for f in day['forecasts']:
                    b=bars.get(f['target_bar'])
                    if b and parse(b[0])+timedelta(minutes=5)<=t:
                        actual=num(b[4])/f['reference_close']-1
                        f.update(actual_return=actual,absolute_error_pct=abs(actual-f['return_15m'])*100,
                                 direction_correct=(actual*f['return_15m']>0))
            except RuntimeError:
                self.diagnostic(day,'forecast_grading','Underlying candles unavailable; forecast accuracy not graded')
        trades=[z for z in self.state['trades'] if z['day']==day['date']]
        lines=[f'NIFTY PAPER REPORT | {day["date"]} | {t:%H:%M} IST',
               'Values/returns are research estimates, not executed trades.']
        for strategy in ('S1','S2','S3'):
            group=[z for z in trades if z['strategy']==strategy and not z['shadow']]
            lines.append(f'\n{strategy}: {len(group)} alerts')
            for z in group:
                item=f'{z["strike"]:g} {z["side"]}: {z["status"]}; entry {z["entry"]:.2f}, last {z["last"]:.2f}'
                if strategy=='S3':
                    item+=f', fixed fair {z["target"]:.2f}, remaining gap {(z["target"]-z["last"])/z["target"]*100:+.1f}%'
                elif 'net_estimated_pct' in z:
                    item+=f', net estimate {z["net_estimated_pct"]:+.2f}%'
                lines.append(item)
        shadows=[z for z in trades if z['shadow'] and z.get('journal_role')!='skipped_candidate']
        skipped=[z for z in trades if z.get('journal_role')=='skipped_candidate']
        lines.append(f'\nSilently followed skipped S1 setups: {len(skipped)}; targets missed {sum(z["status"]=="TARGET" for z in skipped)}; stops avoided {sum(z["status"]=="STOP" for z in skipped)}.')
        graded=[f for f in day.get('forecasts',[]) if 'actual_return' in f]
        if graded:
            lines.append(f'\nKronos underlying forecasts: {sum(f["direction_correct"] for f in graded)}/{len(graded)} correct direction; '
                         f'MAE {statistics.mean(f["absolute_error_pct"] for f in graded):.3f} percentage points.')
        lines.append('\nCounter trades (opposite option, shadow): '+str(len(shadows)))
        for z in shadows:
            lines.append(f'{z["strike"]:g} {z["side"]}: {z["status"]}')
        learner=self.state['learner']; n=len(learner['trained_ids'])
        lines.append(f'\nLearning: {n} resolved S1 samples. '+('Warm-up; no learned filtering yet.' if n<30 else 'Learned ranking active.'))
        if learner['brier']:
            lines.append(f'Forward Brier score: {statistics.mean(learner["brier"]):.4f} (lower is better; uncalibrated small sample).')
        # Associations, not causal explanations.
        for side in ('CE','PE'):
            for bucket in ('normal_iv','high_iv'):
                g=[z for z in self.state['trades'] if z.get('bucket')==f'{side}:{bucket}' and z['status'] in ('TARGET','STOP')]
                if g:
                    wins=sum(z['status']=='TARGET' for z in g)
                    lines.append(f'Observed {side}/{bucket}: {wins}/{len(g)} targets. Association, not proven cause.')
        resolved=[z for z in self.state['trades'] if z['strategy']=='S1' and not z['shadow']
                  and z['status'] in ('TARGET','STOP') and 'x' in z]
        for label,index in [('forecast magnitude',1),('IV/50',2),('absolute Delta',3),('OI change',7),('spread x10',8)]:
            wins=[z['x'][index] for z in resolved if z['status']=='TARGET']
            losses=[z['x'][index] for z in resolved if z['status']=='STOP']
            if wins and losses:
                lines.append(f'{label}: winners mean {statistics.mean(wins):.3f}, losses mean {statistics.mean(losses):.3f}.')
        lines.append('S2 rules and S3 model are never rewritten by the learner. Losses can recur.')
        lines.extend(f'{k}: {v}' for k,v in day['diagnostics'].items())
        review=self.journal.review(day)
        plain=self.journal.english(review)
        text='\n'.join(lines)
        reports=self.dir/'reports';reports.mkdir(exist_ok=True)
        (reports/f'{day["date"]}.txt').write_text(text)
        (reports/f'{day["date"]}.json').write_text(json.dumps({'day':day,'trades':trades,'learner':learner},indent=2))
        (reports/f'{day["date"]}_learning.txt').write_text('\n'.join(plain))
        (reports/'learning_history.json').write_text(json.dumps(self.state['journal'],indent=2))
        if review['milestone']:
            (reports/f'session_{review["session_number"]}_review.txt').write_text('\n'.join(plain))
            (reports/f'session_{review["session_number"]}_review.json').write_text(json.dumps(self.state['journal'],indent=2))
        # Learning summary arrives first; the full results follow.
        lines=plain+['\nDETAILED RESULTS']+lines
        chunks=[];chunk=''
        for line in lines:
            if len(chunk)+len(line)>3500:
                chunks.append(chunk);chunk=''
            chunk+=line+'\n'
        if chunk:chunks.append(chunk)
        for i,chunk in enumerate(chunks):
            self.emit(f'report-{day["date"]}-{i}',chunk)
        day['reported']=True
        self.save()

    def tick(self,t):
        # Recover previous interrupted sessions using historical option candles.
        for date, old in sorted(self.state['days'].items()):
            if date < t.date().isoformat() and not old.get('reported'):
                close=datetime.fromisoformat(date).replace(hour=16,minute=15,tzinfo=IST)
                self.monitor(old,[],close,closing=True)
                self.report(old,close)
        day=self.prepare(t)
        if t.weekday()>=5:
            return
        if t.time()>=dt_time(16,15):
            if not day.get('reported'):
                self.monitor(day,[],t,closing=True)
                self.report(day,t)
            return
        if not dt_time(9,16)<=t.time()<=dt_time(15,35):
            return
        if t.time()>dt_time(15,30):
            self.monitor(day,[],t,closing=True)
            return
        if not self.fresh(INDEX,t):
            self.diagnostic(day,'freshness','No recent completed Nifty candle: holiday, outage or stale feed; entries paused.')
            return
        day['market_seen']=True
        day['diagnostics'].pop('freshness',None)
        chain=self.api.chain(day['expiry'])
        if not chain:
            raise RuntimeError('Empty option chain')
        self.monitor(day,chain,t)
        self.opening_strategy(day,chain,t)
        # Slow model inference is after monitoring and opening alerts; send those first.
        self.flush()
        self.valuation_strategy(day,chain,t)
        self.flush()
        self.learning_strategy(day,chain,t)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['run','once','test'],default='run')
    args=ap.parse_args()
    bot=Bot(API())
    (bot.dir/'reports').mkdir(exist_ok=True)
    if args.mode=='test':
        contracts=bot.api.get('/v2/option/contract',{'instrument_key':INDEX})
        bot.emit('connection-'+stamp(now()),f'Nifty bot connection test OK. Upstox returned {len(contracts)} contracts. Paper-only; all 3 strategies configured. Daily Journal v2 enabled; Kronos unchanged.')
        bot.flush();bot.sync(force=True)
        return
    started=now()
    # Two sessions keep each job comfortably below GitHub's six-hour maximum.
    end=started.replace(hour=14,minute=10,second=0,microsecond=0) if started.time()<dt_time(14,10) else started.replace(hour=16,minute=25,second=0,microsecond=0)
    end=min(end,started+timedelta(hours=5,minutes=15))
    failures=0
    try:
        while True:
            t=now()
            try:
                bot.tick(t);bot.flush();bot.sync()
                failures=0
            except Exception as exc:
                failures+=1
                # Only our own sanitized RuntimeErrors are safe to print.
                reason=str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__
                print(f'{t:%H:%M} tick failed: {reason}',flush=True)
                if failures in (1,10):
                    bot.emit(f'error-{t:%Y%m%d%H}-{failures}',f'Nifty bot data/processing issue: {reason}. No invented signals. Retrying.')
                    try:bot.flush()
                    except Exception:pass
                bot.save()
                if failures>=20:
                    raise RuntimeError('20 consecutive failures; check secrets and provider availability') from None
            if args.mode=='once' or now()>=end or t.weekday()>=5:
                break
            time.sleep(max(5,int(os.getenv('POLL_SECONDS','30'))))
    finally:
        bot.sync(force=True)


if __name__=='__main__':
    main()
