#!/usr/bin/env python3
"""Feature correlation and PCA/t-SNE/UMAP visualization for Phase 2 embeddings."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--hr-labels',default='')
    ap.add_argument('--max-per-case',type=int,default=60)
    args=ap.parse_args(); root=Path(args.root); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    try:
        import umap
        have_umap=True
    except Exception as e:
        have_umap=False; umap_error=repr(e)
    all_summary={"data":"existing VitalDB only","feature_correlations":"descriptive window-level Pearson/Spearman; windows within cases are dependent","visual_sampling":f"up to {args.max_per_case} windows per case", "projections":{}}
    hr_frame=None
    if args.hr_labels and Path(args.hr_labels).exists():
        hf=pd.read_csv(args.hr_labels); hf=hf[hf.target.eq('hr')].set_index('row_i'); hr_frame=hf.label
    for win in (10,30,60):
        sub=root/f'window_{win}s'; idx=pd.read_csv(sub/'window_index.csv'); pack=np.load(sub/'embeddings_papagei.npz'); x=pack['embeddings'].astype(float)
        if len(idx)!=len(x): raise RuntimeError(f'index/cache mismatch at {win}s')
        labels={'SpO2':idx.spo2_median.to_numpy(float)}
        if win==10 and hr_frame is not None:
            labels['HR']=hr_frame.reindex(np.arange(len(idx))).to_numpy(float)
        corr=[]
        for name,y in labels.items():
            ok=np.isfinite(y)
            for j in range(x.shape[1]):
                if ok.sum()<3 or np.std(x[ok,j])==0 or np.std(y[ok])==0: p=s=float('nan')
                else: p=float(pearsonr(x[ok,j],y[ok]).statistic); s=float(spearmanr(x[ok,j],y[ok]).statistic)
                corr.append({'window_s':win,'target':name,'feature':f'feature_{j:03d}','pearson_r':p,'spearman_r':s,'n_windows':int(ok.sum()),'n_cases':int(idx.loc[ok,'case_id'].nunique())})
        cf=pd.DataFrame(corr); cf.to_csv(out/f'feature_correlations_{win}s.csv',index=False)
        tops={}
        for name,g in cf.groupby('target'):
            g=g.assign(abs_pearson=g.pearson_r.abs(),abs_spearman=g.spearman_r.abs())
            tops[name]={'pearson_top10':g.sort_values('abs_pearson',ascending=False).head(10)[['feature','pearson_r']].to_dict('records'),
                        'spearman_top10':g.sort_values('abs_spearman',ascending=False).head(10)[['feature','spearman_r']].to_dict('records')}
        # Keep windows balanced by case to prevent the longest records dominating plots.
        rng=np.random.default_rng(20261011+win); selected=[]
        for cid,grp in idx.groupby('case_id',sort=True):
            ii=grp.index.to_numpy(); selected.extend(rng.choice(ii,min(len(ii),args.max_per_case),replace=False).tolist())
        sel=np.asarray(sorted(selected),dtype=int); xs=x[sel]; ci=idx.iloc[sel].case_id.to_numpy(); ymap={k:v[sel] for k,v in labels.items()}
        pca=PCA(n_components=2,random_state=2026); coords={'pca':pca.fit_transform(xs)}
        coords['tsne']=TSNE(n_components=2,perplexity=min(30,max(5,(len(sel)-1)//3)),init='pca',learning_rate='auto',random_state=2026).fit_transform(xs)
        if have_umap: coords['umap']=umap.UMAP(n_components=2,n_neighbors=20,min_dist=.1,metric='euclidean',random_state=2026).fit_transform(xs)
        for proj,xy in coords.items():
            nplots=3 if 'HR' in ymap else 2; fig,axs=plt.subplots(1,nplots,figsize=(6*nplots,5)); axs=np.atleast_1d(axs)
            panels=[('case_id',ci)] + [(name,y) for name,y in ymap.items()]
            for ax,(name,val) in zip(axs,panels):
                if name=='case_id':
                    code,palette=np.unique(val,return_inverse=True); sc=ax.scatter(xy[:,0],xy[:,1],c=palette,cmap='turbo',s=9,alpha=.75); cb=fig.colorbar(sc,ax=ax); cb.set_ticks(np.arange(len(code))); cb.set_ticklabels(code)
                else: sc=ax.scatter(xy[:,0],xy[:,1],c=val,cmap='viridis',s=9,alpha=.75); fig.colorbar(sc,ax=ax,label=name)
                ax.set_title(f'{proj.upper()} colored by {name}'); ax.set_xlabel('component 1'); ax.set_ylabel('component 2')
            fig.suptitle(f'{win}s PaPaGei embeddings; balanced case sample n={len(sel)}'); fig.tight_layout(); fig.savefig(out/f'{win}s_{proj}_labels.png',dpi=160); plt.close(fig)
        all_summary['projections'][str(win)]={'windows':len(idx),'cases':int(idx.case_id.nunique()),'plotted_windows':len(sel),'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'feature_correlations':tops,'available_projections':list(coords)}
    all_summary['umap_available']=have_umap
    if not have_umap: all_summary['umap_import_error']=umap_error
    (out/'feature_diagnostics_summary.json').write_text(json.dumps(all_summary,indent=2),encoding='utf-8')
    print(json.dumps(all_summary,indent=2))

if __name__=='__main__': main()
