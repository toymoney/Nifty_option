import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'/'Kronos'))
import torch,pandas as pd,numpy as np
from model import Kronos,KronosTokenizer,KronosPredictor
start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42)
tok=KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-2k')
model=Kronos.from_pretrained('NeoQuasar/Kronos-mini')
predictor=KronosPredictor(model,tok,device='cpu',max_context=256)
x=pd.Series(pd.date_range('2026-11-17 09:15',periods=120,freq='5min'))
c=25000+np.arange(120)*.5+np.sin(np.arange(120))
df=pd.DataFrame({'open':c,'high':c+2,'low':c-2,'close':c+1,'volume':np.zeros(120)})
y=pd.Series([x.iloc[-1]+pd.Timedelta(minutes=5*i) for i in [1,2,3]])
pred=predictor.predict(df,x,y,pred_len=3,T=1.,top_p=.9,sample_count=3,verbose=False)
assert len(pred)==3 and np.isfinite(pred['close']).all()
print('Native Kronos CPU synthetic smoke test PASS. Seconds:',round(time.monotonic()-start,1))
print(pred['close'].to_list())
