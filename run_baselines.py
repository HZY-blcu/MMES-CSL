"""Local text-only AES benchmark with GPT reference labels; no external calls.

Adaptation of the character TF-IDF / Ridge / LR / LogisticAT settings exposed
by L2C-rater. This is NOT the full linguistic-feature L2C-rater reproduction.
"""
from pathlib import Path
import sys,json,hashlib,re,unicodedata,time,platform,importlib.util,subprocess
from collections import Counter,defaultdict

TASK=Path(__file__).resolve().parents[1]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'cache/python_packages'))
import numpy as np
import sklearn,scipy,joblib
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score,mean_absolute_error,accuracy_score
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from mord import LogisticAT
import warnings

TRAITS=['内容','结构','词汇','语法']
SEED=20261019
DATA=ROOT/'大模型评分数据集_2000篇_20261015'
BUILD=TASK/'build';BUILD.mkdir(exist_ok=True)
LOCAL=TASK/'data';LOCAL.mkdir(exist_ok=True)
REPOS=TASK/'source/repos'

def read_jsonl(p):return [json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
def write_json(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def write_jsonl(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def normalized(text):return re.sub(r'\s+','',unicodedata.normalize('NFKC',text))
def safe_qwk(y,p):
    # Fixed ordinal labels 1..5 for all models and all slices.
    o=np.bincount((np.asarray(y,dtype=int)-1)*5+np.asarray(p,dtype=int)-1,minlength=25).reshape(5,5)
    w=(np.arange(5)[:,None]-np.arange(5)[None,:])**2
    e=np.outer(o.sum(1),o.sum(0))/o.sum()
    d=float((w*e).sum())
    return None if d==0 else float(1-(w*o).sum()/d)
def metrics(y,p):
    return {'qwk':safe_qwk(y,p),'mae':float(mean_absolute_error(y,p)),'accuracy':float(accuracy_score(y,p)),'within_one':float(np.mean(np.abs(y-p)<=1)),'n':len(y)}

def main():
    start=time.time()
    raw=sorted(read_jsonl(DATA/'模型评分材料/模型输入.jsonl'),key=lambda r:r['essay_id'])
    manifest={r['essay_id']:r for r in read_jsonl(DATA/'研究者资料/抽样清单.jsonl')}
    score_files=sorted((DATA/'评分结果/Codex_独立评分_20261015/CLI分片/scores').glob('*.json'))
    labels={}
    for p in score_files:
        r=json.loads(p.read_text(encoding='utf-8-sig'));assert r['essay_id'] not in labels;labels[r['essay_id']]=r
    assert len(raw)==2000 and {r['essay_id'] for r in raw}==set(labels)==set(manifest)
    for r in raw:
        ref=labels[r['essay_id']]
        assert set(ref['scores'])==set(TRAITS+['书写'])
        assert all(isinstance(v,int) and not isinstance(v,bool) and 1<=v<=5 for v in ref['scores'].values())
        assert hashlib.sha256((DATA/'模型评分材料'/r['text_path']).read_bytes()).hexdigest()==manifest[r['essay_id']]['text_file_sha256']
        assert len(ref['image_observations'])==len(r['image_paths'])
    texts=[r['text'] for r in raw]
    norm=[normalized(t) for t in texts]
    exact=defaultdict(list)
    for i,t in enumerate(norm):exact[hashlib.sha256(t.encode()).hexdigest()].append(i)
    parent=list(range(len(raw)))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):parent[find(b)]=find(a)
    for ids in exact.values():
        for i in ids[1:]:union(ids[0],i)
    # Unsupervised corpus-wide duplicate screening; these features never enter a predictor.
    dup=TfidfVectorizer(analyzer='char',ngram_range=(3,5),dtype=np.float32).fit_transform(norm)
    sim=(dup@dup.T).tocoo()
    candidates=[(int(i),int(j),float(v)) for i,j,v in zip(sim.row,sim.col,sim.data) if i<j and v>=0.90]
    shingles=[set(t[k:k+5] for k in range(max(1,len(t)-4))) for t in norm]
    near=[]
    for i,j,c in candidates:
        jac=len(shingles[i]&shingles[j])/len(shingles[i]|shingles[j])
        if jac>=0.85:union(i,j);near.append({'a':raw[i]['essay_id'],'b':raw[j]['essay_id'],'cosine':c,'jaccard':jac})
    groups=[find(i) for i in range(len(raw))]
    strata=[f"{r['source_group']}:{manifest[r['essay_id']]['band_index']}" for r in raw]
    if len(set(groups))==len(raw):
        trainval,test=train_test_split(np.arange(len(raw)),test_size=0.2,random_state=SEED,stratify=strata)
        train,valid=train_test_split(trainval,test_size=0.125,random_state=SEED,stratify=[strata[i] for i in trainval])
    else:
        folds=list(StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=SEED).split(np.zeros(len(raw)),strata,groups))
        test=np.sort(np.concatenate([folds[0][1],folds[1][1]]));valid=np.sort(folds[2][1]);train=np.sort(np.concatenate([f[1] for f in folds[3:]]))
    splits={'train':sorted(map(int,train)),'validation':sorted(map(int,valid)),'test':sorted(map(int,test))}
    assert len(set(sum(splits.values(),[])))==2000
    assert not set(groups[i] for i in train)&set(groups[i] for i in test)
    assert not set(groups[i] for i in valid)&set(groups[i] for i in test)
    identities={n:[raw[i]['essay_id'] for i in ids] for n,ids in splits.items()}
    fingerprint=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()
    fixed=LOCAL/'splits.json'
    if fixed.exists():assert json.loads(fixed.read_text(encoding='utf-8'))['split_sha256']==fingerprint,'Refuse changing an established split'
    write_json(fixed,{'seed':SEED,'split_sha256':fingerprint,'stratification':'source_group x original_score_band','near_duplicate_rule':'char3-5 TF-IDF cosine >=0.90 AND char5 Jaccard >=0.85; duplicate groups stay together','splits':identities})
    rows=[]
    for r in raw:
        m=manifest[r['essay_id']];ref=labels[r['essay_id']]
        rows.append({'essay_id':r['essay_id'],'title':r['title'],'text':r['text'],'image_paths':r['image_paths'],'scores':ref['scores'],'label_source':'GPT-6.1-Sol','scoring_metadata':ref['scoring_metadata']})
    for name,ids in splits.items():
        write_jsonl(LOCAL/f'{name}.jsonl',[rows[i] for i in ids])
        write_jsonl(LOCAL/f'{name}_inputs.jsonl',[{k:rows[i][k] for k in ['essay_id','title','text','image_paths']} for i in ids])
    write_json(BUILD/'data_audit.json',{'total':2000,'split_sizes':{n:len(ids) for n,ids in splits.items()},'split_sha256':fingerprint,'exact_duplicate_groups':[[raw[i]['essay_id'] for i in v] for v in exact.values() if len(v)>1],'near_duplicates':near,'distinct_titles':len({r['title'] for r in raw}),'author_identity':'Essay IDs are not assumed to identify writers; writer-level overlap remains unknown.','distribution':{n:{g:sum(raw[i]['source_group']==g for i in ids) for g in ['ji_xu','yi_lun','ying_yong','hsk_zuowen']} for n,ids in splits.items()},'scoring_effort':{n:dict(Counter(str(rows[i]['scoring_metadata'].get('reasoning_effort','unknown')) for i in ids)) for n,ids in splits.items()}})
    textmod=load_module('l2c_text_repr',REPOS/'L2C-rater/utils/text_repre.py')
    authqwk=load_module('l2c_qwk',REPOS/'L2C-rater/quadratic_weighted_kappa.py').quadratic_weighted_kappa
    corpus=[textmod.get_text_feature_from_ltp_results(t,{},feature='c') for t in texts]
    vectorizer=CountVectorizer(ngram_range=(1,1),min_df=10,token_pattern=r'[\w，,。.！!？?；;、]+')
    tfidf=TfidfTransformer()
    xtrain=tfidf.fit_transform(vectorizer.fit_transform([corpus[i] for i in train]))
    xvalid=tfidf.transform(vectorizer.transform([corpus[i] for i in valid]));xtest=tfidf.transform(vectorizer.transform([corpus[i] for i in test]))
    author_x=textmod.get_text_matrix([corpus[i] for i in train],ngram_min=1,ngram_max=1,df_threshold=10)
    assert author_x.shape==xtrain.shape and (author_x-xtrain).nnz==0
    joblib.dump({'vectorizer':vectorizer,'tfidf':tfidf},BUILD/'character_tfidf.joblib')
    y=np.array([[r['scores'][t] for t in TRAITS] for r in rows],dtype=int)
    predictions={};results={};tuning={};models={};convergence=[]
    for method in ['majority','ridge','logistic','ordinal_logistic']:
        predictions[method]=np.zeros((len(test),4),dtype=int);results[method]={};tuning[method]={};models[method]={}
        for k,trait in enumerate(TRAITS):
            if method=='majority':
                value=Counter(y[train,k]).most_common(1)[0][0]
                pred=np.full(len(test),value);tuning[method][trait]={'train_mode':int(value)}
            elif '--summarize' in sys.argv:
                model=joblib.load(BUILD/f'{method}.joblib')[trait]
                pv=np.rint(np.clip(model.predict(xvalid),1,5)).astype(int)
                pred=np.rint(np.clip(model.predict(xtest),1,5)).astype(int)
                saved={r['essay_id']:r['scores'][trait] for r in read_jsonl(BUILD/f'{method}_test_predictions.jsonl')}
                assert [saved[raw[i]['essay_id']] for i in test]==pred.tolist()
                key='C' if method=='logistic' else 'alpha'
                tuning[method][trait]={'parameter':float(getattr(model,key)),'parameter_name':key,'validation':metrics(y[valid,k],pv)}
                models[method][trait]=model
            else:
                best=None
                for a in [0.1,1.,10.]:
                    model=Ridge(alpha=a,solver='lsqr') if method=='ridge' else LogisticRegression(C=a,max_iter=1000,random_state=SEED) if method=='logistic' else LogisticAT(alpha=a,max_iter=1000)
                    with warnings.catch_warnings(record=True) as observed:
                        warnings.simplefilter('always');model.fit(xtrain,y[train,k])
                    for warning in observed:convergence.append({'method':method,'trait':trait,'parameter':a,'type':warning.category.__name__,'message':str(warning.message)})
                    pv=np.rint(np.clip(model.predict(xvalid),1,5)).astype(int)
                    score=safe_qwk(y[valid,k],pv)
                    key=(score if score is not None else -float('inf'),-mean_absolute_error(y[valid,k],pv))
                    if best is None or key>best[0]:best=(key,a,model,metrics(y[valid,k],pv))
                model=best[2];pred=np.rint(np.clip(model.predict(xtest),1,5)).astype(int)
                tuning[method][trait]={'parameter':best[1],'parameter_name':'C' if method=='logistic' else 'alpha','validation':best[3]};models[method][trait]=model
            predictions[method][:,k]=pred
            result=metrics(y[test,k],pred)
            assert abs(result['qwk']-authqwk(y[test,k],pred,1,5))<1e-12
            assert abs(result['qwk']-cohen_kappa_score(y[test,k],pred,labels=[1,2,3,4,5],weights='quadratic'))<1e-12
            results[method][trait]=result
        results[method]['macro_qwk']=float(np.mean([results[method][t]['qwk'] for t in TRAITS]))
        if models[method]:joblib.dump(models[method],BUILD/f'{method}.joblib')
        write_jsonl(BUILD/f'{method}_test_predictions.jsonl',[{'essay_id':raw[i]['essay_id'],'scores':{t:int(predictions[method][j,k]) for k,t in enumerate(TRAITS)}} for j,i in enumerate(test)])
        print(method,json.dumps(results[method],ensure_ascii=False),flush=True)
    rng=np.random.default_rng(SEED)
    boot=rng.integers(0,len(test),size=(1000,len(test)))
    for method,pred in predictions.items():
        values=[]
        for idx in boot:
            qs=[safe_qwk(y[test[idx],k],pred[idx,k]) for k in range(4)]
            if all(q is not None for q in qs):values.append(np.mean(qs))
        results[method]['macro_qwk_bootstrap_95ci']=list(map(float,np.quantile(values,[0.025,0.975])))
        results[method]['by_source_group']={}
        for g in ['ji_xu','yi_lun','ying_yong','hsk_zuowen']:
            mask=np.array([raw[i]['source_group']==g for i in test])
            results[method]['by_source_group'][g]={t:metrics(y[test[mask],k],pred[mask,k]) for k,t in enumerate(TRAITS)}
    rev={name:subprocess.check_output(['git','-c',f'safe.directory={(REPOS/name).as_posix()}','-C',str(REPOS/name),'rev-parse','HEAD'],text=True).strip() for name in ['L2C-rater','CTS','RMTS','EssayJudge','MTS']}
    write_json(BUILD/'results.json',{'reference':'GPT-generated ordinal scores; not independent human gold','scope':'Text-only four-trait adaptation of L2C-rater character TF-IDF settings; not full paper reproduction','seed':SEED,'train_n':len(train),'validation_n':len(valid),'test_n':len(test),'vocabulary_size':xtrain.shape[1],'selection':'Per-trait hyperparameters selected on validation QWK, MAE breaks ties. No test tuning. Training is not refit on validation.','rounding':'np.rint after clipping 1..5; fixed before test','parameter_grid':[0.1,1.,10.],'split_sha256':fingerprint,'repository_revisions':rev,'python':platform.python_version(),'packages':{'sklearn':sklearn.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'mord':'0.7','jieba':'0.42.1'},'author_character_matrix_exact_match':True,'qwk_crosschecked_against_author_and_sklearn':True,'results':results,'tuning':tuning,'warnings':convergence,'seconds':time.time()-start})
    print('saved',str(BUILD/'results.json'),flush=True)

if __name__=='__main__':main()
