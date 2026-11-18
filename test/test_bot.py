import copy
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from nifty_options_bot import Bot, IST, atm, bs, peer_iv, candle_hit, features, probability

T=datetime(2026,11,18,10,0,15,tzinfo=IST)

def contract(strike=25650,side='CE',price=100,iv=20):
    return dict(key=f'{strike}{side}',strike=strike,side=side,expiry='2026-11-24',spot=25626,
        market=dict(ltp=price,bid_price=price-.5,ask_price=price+.5,oi=10000,prev_oi=9000,volume=500),
        greeks=dict(iv=iv,delta=.5,gamma=.001,theta=-10,vega=20))

def bar(t,o,h,l,c):return [t.isoformat(),o,h,l,c,100,10000]

class FakeAPI:
    def __init__(self, bars=None):self.bars=bars or []
    def candles(self,*args,**kwargs):return self.bars

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.bot=Bot(FakeAPI(), self.tmp.name)
        self.day={'date':T.date().isoformat(),'diagnostics':{}}
    def tearDown(self):
        self.bot.executor.shutdown(wait=False,cancel_futures=True)
        self.tmp.cleanup()
    def trade(self,strategy='S2',extra=None):
        return self.bot.create_trade(strategy,contract(),T,self.day,extra or
            ({'previous_close':95,'opening':101} if strategy=='S2' else {}))
    def test_atm_rounding(self):
        self.assertEqual(atm(25626),25650)
        self.assertEqual(atm(25624),25600)
    def test_put_call_parity(self):
        import math
        c=bs(100,100,1,.2,'CE',.05,.01);p=bs(100,100,1,.2,'PE',.05,.01)
        self.assertAlmostEqual(c-p,100*math.exp(-.01)-100*math.exp(-.05))
    def test_own_iv_excluded(self):
        c=contract(); peers=[contract(x) for x in (25550,25600,25700,25750)]
        a=peer_iv(c,peers+[c]);c['greeks']['iv']=100
        self.assertAlmostEqual(a,peer_iv(c,peers+[c]));self.assertAlmostEqual(a,.2)
    def test_no_extrapolation(self):
        self.assertIsNone(peer_iv(contract(),[contract(25700),contract(25750)]))
    def test_fixed_levels(self):
        z=self.trade();self.assertEqual(z['target'],115);self.assertEqual(z['stop'],90)
    def test_duplicate_entry(self):
        self.trade();self.assertIsNone(self.trade());self.assertEqual(len(self.bot.state['outbox']),1)
    def test_both_touches_ambiguous(self):
        z=self.trade();self.assertEqual(candle_hit(z,bar(T,100,116,89,100)),'AMBIGUOUS')
    def test_target_only(self):
        z=self.trade();self.assertEqual(candle_hit(z,bar(T,100,116,91,110)),'TARGET')
    def test_down_fair_value(self):
        z=self.trade('S3',dict(target=80,stop=None,direction='DOWN',gap_pct=-25,peer_iv=20))
        self.assertEqual(candle_hit(z,bar(T,100,101,79,81)),'VALUE_REACHED')
    def test_entry_minute_excluded(self):
        z=self.trade();self.bot.api.bars=[bar(T.replace(second=0),100,120,80,100)]
        self.bot.monitor(self.day,[],T+timedelta(minutes=2))
        self.assertEqual(z['status'],'OPEN')
    def test_chronological_first_touch(self):
        z=self.trade();self.bot.api.bars=[bar(T.replace(minute=1,second=0),100,114,89,95),bar(T.replace(minute=2,second=0),95,120,95,119)]
        self.bot.monitor(self.day,[],T+timedelta(minutes=4))
        self.assertEqual(z['status'],'STOP')
        self.assertEqual(len(self.bot.state['outbox']),2)
        self.bot.monitor(self.day,[],T+timedelta(minutes=5))
        self.assertEqual(len(self.bot.state['outbox']),2)
    def test_gap_stop_fill(self):
        z=self.trade();self.bot.api.bars=[bar(T.replace(minute=1,second=0),85,89,80,86)]
        self.bot.monitor(self.day,[],T+timedelta(minutes=3))
        self.assertEqual(z['exit_price'],85)
    def test_restart_preserves_dedup(self):
        self.trade();other=Bot(FakeAPI(),self.tmp.name)
        self.assertEqual(len(other.state['trades']),1)
        self.assertIsNone(other.create_trade('S2',contract(),T,self.day))
        other.executor.shutdown(wait=False)
    def test_training_only_resolved_main_s1(self):
        x=features(contract(),.002,T)
        main=self.trade('S1',dict(x=x,score_at_entry=.5))
        self.bot.finish(main,'TARGET',T,115)
        self.bot.train(self.day)
        self.assertEqual(len(self.bot.state['learner']['trained_ids']),1)
        weights=list(self.bot.state['learner']['weights'])
        self.assertGreater(probability(weights,x),.5)
        self.bot.train(self.day)
        self.assertEqual(weights,self.bot.state['learner']['weights'])
    def test_ambiguous_not_trained(self):
        z=self.trade('S1',dict(x=features(contract(),.002,T),score_at_entry=.5))
        self.bot.finish(z,'AMBIGUOUS',T)
        self.bot.train(self.day)
        self.assertEqual(self.bot.state['learner']['trained_ids'],[])
    def test_shadow_not_trained(self):
        z=self.bot.create_trade('S1',contract(),T,self.day,{'x':[1]*10,'score_at_entry':.5},shadow=True)
        self.bot.finish(z,'STOP',T,90);self.bot.train(self.day)
        self.assertEqual(self.bot.state['learner']['trained_ids'],[])
        self.assertEqual(self.bot.state['outbox'],[])
    def test_late_open_skipped(self):
        self.bot.opening_strategy(self.day,[],T)
        self.assertTrue(self.day['opening_done']);self.assertEqual(self.bot.state['trades'],[])
    def test_report_once_and_fixed_value(self):
        z=self.trade('S3',dict(target=120,stop=None,direction='UP',gap_pct=16.67,peer_iv=20))
        self.bot.report(self.day,T);n=len(self.bot.state['outbox'])
        self.bot.report(self.day,T);self.assertEqual(n,len(self.bot.state['outbox']))
        self.assertEqual(z['target'],120)
    def test_missing_data_not_loss(self):
        z=self.trade();self.bot.monitor(self.day,[],T.replace(hour=16),closing=True)
        self.assertEqual(z['status'],'DATA_MISSING')
    def test_telegram_failure_keeps_outbox(self):
        self.trade()
        import requests
        with patch.dict('os.environ',{'TELEGRAM_BOT_TOKEN':'secret','TELEGRAM_CHAT_ID':'id'}),patch('requests.post',side_effect=requests.Timeout('secret')):
            with self.assertRaisesRegex(RuntimeError,'Telegram network'):
                self.bot.flush()
        self.assertFalse(self.bot.state['outbox'][0]['sent'])

if __name__=='__main__':unittest.main()

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.bot=Bot(FakeAPI(),self.tmp.name)
    def tearDown(self):
        self.bot.executor.shutdown(wait=False,cancel_futures=True);self.tmp.cleanup()
    def test_opening_two_sides_same_contract_prior_session(self):
        t=T.replace(hour=9,minute=17)
        day={'date':t.date().isoformat(),'previous_date':'2026-11-17',
             'opening_contracts':{'CE':'25650CE','PE':'25650PE'},'diagnostics':{}}
        class OpeningAPI:
            def candles(self,key,t,**kwargs):
                if kwargs.get('history'):
                    return [bar(t.replace(day=17,hour=0,minute=0),95,100,90,95)]
                return [bar(t.replace(hour=9,minute=15,second=0),101,102,100,101),
                        bar(t.replace(hour=9,minute=16,second=0),101,102,100,101)]
        self.bot.api=OpeningAPI()
        self.bot.opening_strategy(day,[contract(),contract(side='PE')],t)
        self.assertEqual(len(self.bot.state['trades']),2)
        for z in self.bot.state['trades']:
            self.assertEqual(z['previous_close'],95);self.assertEqual(z['opening'],101)
            self.assertEqual(z['target'],115)
        self.bot.opening_strategy(day,[contract(),contract(side='PE')],t)
        self.assertEqual(len(self.bot.state['trades']),2)
    def test_snapshot_target_waits_for_completed_candle(self):
        day={'date':T.date().isoformat(),'diagnostics':{}}
        z=self.bot.create_trade('S2',contract(),T,day,{'previous_close':95,'opening':101})
        self.bot.api.bars=[bar(T.replace(second=0),100,100,100,100)]
        with patch.object(self.bot,'fresh',return_value=True):
            self.bot.monitor(day,[contract(price=116)],T+timedelta(seconds=30))
        self.assertEqual(z['status'],'OPEN')
    def test_valuation_only_within_five_strikes(self):
        day={'date':T.date().isoformat(),'expiry':'2026-11-24','diagnostics':{}}
        chain=[contract(k,side,price=20) for k in range(25000,26301,50) for side in ('CE','PE')]
        with patch.object(self.bot,'fresh',return_value=True):
            self.bot.valuation_strategy(day,chain,T)
        self.assertTrue(self.bot.state['trades'])
        self.assertTrue(all(abs(z['strike']-25650)<=250 for z in self.bot.state['trades']))
        self.assertTrue(all(z['target']!=round(z['entry']*1.15,2) for z in self.bot.state['trades']))
    def test_monitor_failure_is_missing_not_open_forever(self):
        day={'date':T.date().isoformat(),'diagnostics':{}}
        z=self.bot.create_trade('S2',contract(),T,day,{'previous_close':95,'opening':101})
        with patch.object(self.bot.api,'candles',side_effect=RuntimeError('unavailable')):
            self.bot.monitor(day,[],T.replace(hour=16),closing=True)
        self.assertEqual(z['status'],'DATA_MISSING')
    def test_native_forecast_result_creates_main_and_counter(self):
        from concurrent.futures import Future
        day={'date':T.date().isoformat(),'diagnostics':{}}
        future=Future()
        future.set_result({'return_15m':.002,'reference_close':25000,'predicted_close':25050,
                           'reference_bar':T.isoformat(),'target_bar':(T+timedelta(minutes=15)).isoformat()})
        self.bot.pending_forecast=future;self.bot.forecast_started=T-timedelta(seconds=30)
        with patch.object(self.bot,'fresh',return_value=True):
            self.bot.learning_strategy(day,[contract(),contract(side='PE')],T)
        self.assertEqual(len(self.bot.state['trades']),2)
        main,counter=self.bot.state['trades']
        self.assertFalse(main['shadow']);self.assertEqual(main['side'],'CE')
        self.assertTrue(counter['shadow']);self.assertEqual(counter['side'],'PE')
        self.assertEqual(counter['parent_id'],main['id'])
    def test_kronos_forward_forecast_grading(self):
        target=T.replace(minute=15,second=0)
        day={'date':T.date().isoformat(),'diagnostics':{},'forecasts':[
            {'reference_close':25000,'return_15m':.002,'target_bar':target.isoformat()}]}
        self.bot.api.bars=[bar(target,25000,25060,24990,25025)]
        self.bot.report(day,T.replace(hour=16))
        f=day['forecasts'][0]
        self.assertTrue(f['direction_correct'])
        self.assertAlmostEqual(f['absolute_error_pct'],.1)
