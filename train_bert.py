"""Full Chinese BERT multi-output fine-tuning on the fixed GPT-labeled split.

All essay tokens are included via overlapping chunks. Model selection uses
validation macro QWK only. Evaluate the fixed test set once after selection.
"""
from pathlib import Path
import os,sys,json,math,random,time,hashlib,argparse,platform
from collections import Counter

TASK=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer,AutoModel,get_linear_schedule_with_warmup
from run_baselines import TRAITS,read_jsonl,write_json,write_jsonl,metrics,safe_qwk
import numpy as np

MODEL=TASK/'cache/models/bert-base-chinese'
OUT=TASK/'build/bert_base_chinese'
CONFIG={
    'model_id':'google-bert/bert-base-chinese',
    'revision':'8f23c25b06e129b6c986331a13d8d025a92cf0ea',
    'seed':20261019,'max_epochs':10,'patience':3,'learning_rate':2e-5,
    'batch_size_documents':4,'gradient_accumulation':4,'weight_decay':0.01,
    'warmup_ratio':0.1,'max_gradient_norm':1.0,'content_tokens_per_chunk':510,
    'chunk_start_step':384,'pooling':'unweighted mean of final-layer CLS across all document chunks',
    'head':'dropout(0.1), linear(hidden_size,4), 1+4*sigmoid',
    'loss':'mean squared error over all four traits and documents; scores normalized to [0,1]',
    'optimizer':'AdamW; bias and LayerNorm weight excluded from weight decay',
    'selection':'best validation macro QWK, then lower validation MAE; no test tuning',
    'inference_rounding':'clip [1,5] then np.rint; fixed before training',
    'input':'题目 + full student text; no reference scores, rationales or scoring metadata',
    'gradient_checkpointing':True,'gpu_memory_fraction_limit':0.60,
}

class EssayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(MODEL,local_files_only=True,use_safetensors=True,add_pooling_layer=False,attn_implementation='sdpa')
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.dropout=nn.Dropout(0.1)
        self.head=nn.Linear(self.encoder.config.hidden_size,4)
    def forward(self,encoded,chunk_counts):
        cls=self.encoder(**encoded).last_hidden_state[:,0,:]
        pooled=[];start=0
        for count in chunk_counts:
            pooled.append(cls[start:start+count].mean(0));start+=count
        assert start==len(cls)
        return 1+4*torch.sigmoid(self.head(self.dropout(torch.stack(pooled))))

def tokenize(rows,tokenizer):
    out=[];audit=[]
    for row in rows:
        text='题目：'+row['title']+'\n作文全文：\n'+row['text']
        tokens=tokenizer.encode(text,add_special_tokens=False,truncation=False)
        chunks=[];covered=set();start=0
        while start<len(tokens):
            end=min(len(tokens),start+CONFIG['content_tokens_per_chunk'])
            chunk=[tokenizer.cls_token_id]+tokens[start:end]+[tokenizer.sep_token_id]
            chunks.append({'input_ids':chunk,'attention_mask':[1]*len(chunk),'token_type_ids':[0]*len(chunk)})
            covered.update(range(start,end))
            if end==len(tokens):break
            start+=CONFIG['chunk_start_step']
        assert len(covered)==len(tokens) and all(len(c['input_ids'])<=512 for c in chunks)
        out.append({'essay_id':row['essay_id'],'chunks':chunks,'labels':[row['scores'][t] for t in TRAITS]})
        audit.append({'essay_id':row['essay_id'],'tokens':len(tokens),'chunks':len(chunks),'tokens_covered':len(covered)})
    return out,audit

def evaluate(model,loader,device,dtype):
    model.eval();pred=[];labels=[];ids=[]
    with torch.inference_mode():
        for batch in loader:
            encoded={k:v.to(device) for k,v in batch['encoded'].items()}
            with torch.autocast('cuda',dtype=dtype):scores=model(encoded,batch['chunk_counts'])
            pred.extend(scores.float().cpu().tolist());labels.extend(batch['labels'].tolist());ids.extend(batch['essay_ids'])
    continuous=np.array(pred);gold=np.array(labels,dtype=int);rounded=np.rint(np.clip(continuous,1,5)).astype(int)
    result={t:metrics(gold[:,k],rounded[:,k]) for k,t in enumerate(TRAITS)}
    qs=[result[t]['qwk'] for t in TRAITS]
    result['macro_qwk']=float(np.mean(qs)) if all(q is not None for q in qs) else None
    result['macro_mae']=float(np.mean([result[t]['mae'] for t in TRAITS]))
    return result,ids,continuous,rounded,gold

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--evaluate-only',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available(),'CUDA required for this configured experiment'
    if (OUT/'results.json').exists() and not a.evaluate_only:raise RuntimeError('Completed experiment exists; refusing accidental retraining')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_memory_fraction_limit'])
    random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    split=json.loads((TASK/'data/splits.json').read_text(encoding='utf-8'))
    weight_hash=hashlib.sha256((MODEL/'model.safetensors').read_bytes()).hexdigest()
    assert weight_hash=='3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3'
    config={**CONFIG,'split_sha256':split['split_sha256'],'pretrained_weight_sha256':weight_hash,'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'precision':str(dtype)}
    existing=OUT/'configuration.json'
    if existing.exists():assert json.loads(existing.read_text(encoding='utf-8'))==config,'Configuration changed'
    else:write_json(existing,config)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tokenized={};audit={}
    for name in ['train','validation','test']:
        rows=read_jsonl(TASK/f'data/{name}.jsonl')
        assert [r['essay_id'] for r in rows]==split['splits'][name]
        tokenized[name],audit[name]=tokenize(rows,tokenizer)
    write_json(OUT/'token_coverage.json',audit)
    def collate(rows):
        chunks=[chunk for row in rows for chunk in row['chunks']]
        encoded=tokenizer.pad(chunks,padding=True,pad_to_multiple_of=8,return_tensors='pt')
        return {'encoded':encoded,'chunk_counts':[len(r['chunks']) for r in rows],'labels':torch.tensor([r['labels'] for r in rows],dtype=torch.float32),'essay_ids':[r['essay_id'] for r in rows]}
    loaders={name:DataLoader(rows,batch_size=CONFIG['batch_size_documents'],shuffle=name=='train',collate_fn=collate,num_workers=0) for name,rows in tokenized.items()}
    model=EssayModel().to(device)
    config['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert all(p.requires_grad for p in model.encoder.parameters())
    groups=[{'params':[v for n,v in model.named_parameters() if 'bias' not in n and 'LayerNorm.weight' not in n],'weight_decay':CONFIG['weight_decay']},{'params':[v for n,v in model.named_parameters() if 'bias' in n or 'LayerNorm.weight' in n],'weight_decay':0.0}]
    optimizer=torch.optim.AdamW(groups,lr=CONFIG['learning_rate'])
    updates_per_epoch=math.ceil(len(loaders['train'])/CONFIG['gradient_accumulation'])
    total=updates_per_epoch*CONFIG['max_epochs']
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(CONFIG['warmup_ratio']*total),num_training_steps=total)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    epoch_start=0;best_key=(-float('inf'),-float('inf'));best_epoch=None;stale=0;history=[]
    if a.resume:
        state=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        epoch_start=state['epoch'];best_key=state['best_key'];best_epoch=state['best_epoch'];stale=state['stale'];history=state['history']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    started=time.time()
    if not a.evaluate_only:
        for epoch in range(epoch_start,CONFIG['max_epochs']):
            model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;seen=0;ep_started=time.time()
            for step,batch in enumerate(loaders['train']):
                encoded={k:v.to(device) for k,v in batch['encoded'].items()};target=batch['labels'].to(device)
                accumulation=CONFIG['gradient_accumulation']
                group_size=min(accumulation,len(loaders['train'])-(step//accumulation)*accumulation)
                with torch.autocast('cuda',dtype=dtype):
                    predicted=model(encoded,batch['chunk_counts'])
                    mse=nn.functional.mse_loss((predicted-1)/4,(target-1)/4)
                    loss=mse/group_size
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                scaler.scale(loss).backward();loss_sum+=float(mse.detach())*len(target);seen+=len(target)
                if (step+1)%accumulation==0 or step+1==len(loaders['train']):
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['max_gradient_norm'])
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);scheduler.step()
                if (step+1)%50==0:
                    status={'status':'training','epoch':epoch+1,'max_epochs':CONFIG['max_epochs'],'documents_seen_in_epoch':seen,'train_documents':1400,'normalized_mse':loss_sum/seen,'elapsed_seconds':time.time()-started}
                    write_json(OUT/'progress.json',status);print(json.dumps(status,ensure_ascii=False),flush=True)
            result,_,_,_,_=evaluate(model,loaders['validation'],device,dtype)
            key=(result['macro_qwk'] if result['macro_qwk'] is not None else -float('inf'),-result['macro_mae'])
            entry={'epoch':epoch+1,'train_normalized_mse':loss_sum/seen,'validation':result,'seconds':time.time()-ep_started};history.append(entry)
            if key>best_key:
                best_key=key;best_epoch=epoch+1;stale=0
                torch.save(model.state_dict(),OUT/'best.pt')
            else:stale+=1
            write_json(OUT/'training_history.json',history)
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,'best_key':best_key,'best_epoch':best_epoch,'stale':stale,'history':history,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},OUT/'last.pt')
            write_json(OUT/'progress.json',{'status':'epoch_completed','epoch':epoch+1,'best_epoch':best_epoch,'validation_macro_qwk':result['macro_qwk'],'best_validation_macro_qwk':best_key[0],'stale_epochs':stale})
            print('EPOCH_COMPLETE',json.dumps(entry,ensure_ascii=False),flush=True)
            if stale>=CONFIG['patience']:break
    else:
        history=json.loads((OUT/'training_history.json').read_text(encoding='utf-8'));best_epoch=max(history,key=lambda r:(r['validation']['macro_qwk'] if r['validation']['macro_qwk'] is not None else -float('inf'),-r['validation']['macro_mae']))['epoch']
    model.load_state_dict(torch.load(OUT/'best.pt',map_location=device,weights_only=True))
    result,ids,continuous,rounded,gold=evaluate(model,loaders['test'],device,dtype)
    assert len(ids)==400 and len(set(ids))==400
    write_jsonl(OUT/'test_predictions.jsonl',[{'essay_id':eid,'scores':{t:int(rounded[j,k]) for k,t in enumerate(TRAITS)},'continuous_scores':{t:float(continuous[j,k]) for k,t in enumerate(TRAITS)}} for j,eid in enumerate(ids)])
    source={r['essay_id']:r['source_group'] for r in read_jsonl(TASK.parents[2]/'大模型评分数据集_2000篇_20261015/研究者资料/抽样清单.jsonl')}
    by_source={}
    for group in ['ji_xu','yi_lun','ying_yong','hsk_zuowen']:
        mask=np.array([source[eid]==group for eid in ids]);by_source[group]={t:metrics(gold[mask,k],rounded[mask,k]) for k,t in enumerate(TRAITS)}
    rng=np.random.default_rng(CONFIG['seed']);samples=rng.integers(0,400,size=(1000,400));bootstrap=[];diff=[]
    baseline={r['essay_id']:r['scores'] for r in read_jsonl(TASK/'build/ordinal_logistic_test_predictions.jsonl')}
    baseline_array=np.array([[baseline[eid][t] for t in TRAITS] for eid in ids])
    for idx in samples:
        q=[safe_qwk(gold[idx,k],rounded[idx,k]) for k in range(4)];b=[safe_qwk(gold[idx,k],baseline_array[idx,k]) for k in range(4)]
        if all(v is not None for v in q+b):bootstrap.append(np.mean(q));diff.append(np.mean(q)-np.mean(b))
    final={'reference':'GPT scores','scope':'Chinese BERT full-encoder multi-output regression, text-only four traits','configuration':config,'best_epoch':best_epoch,'completed_epochs':len(history),'train_n':1400,'validation_n':200,'test_n':400,'test':result,'macro_qwk_bootstrap_95ci':list(map(float,np.quantile(bootstrap,[.025,.975]))),'paired_macro_qwk_difference_vs_ordinal_logistic_95ci':list(map(float,np.quantile(diff,[.025,.975]))),'by_source_group':by_source,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'elapsed_this_invocation_seconds':time.time()-started,'checkpoint_sha256':hashlib.sha256((OUT/'best.pt').read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(OUT/'results.json',final);write_json(OUT/'progress.json',{'status':'completed','best_epoch':best_epoch,'completed_epochs':len(history),'test_macro_qwk':result['macro_qwk']})
    print('TEST_COMPLETE',json.dumps(final,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
