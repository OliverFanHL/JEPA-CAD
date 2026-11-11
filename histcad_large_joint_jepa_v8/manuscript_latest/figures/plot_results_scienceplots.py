"""Compact vector figures from the saved v8 evaluation outputs (SciencePlots/Arial)."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/".plot_deps"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import scienceplots  # noqa: F401
plt.style.use(["science","no-latex"])
plt.rcParams.update({"font.family":"Arial","font.sans-serif":["Arial"],
    "font.size":8.5,"axes.labelsize":9,"xtick.labelsize":8.1,
    "ytick.labelsize":8.4,"legend.fontsize":8.3,"pdf.fonttype":42,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,
    "xtick.direction":"out","ytick.direction":"out",
    "xtick.minor.visible":False,"ytick.minor.visible":False,
    "xtick.top":False,"ytick.right":False})
BLUE,ORANGE,GREEN="#456778","#a66d4b","#668478"
GRAY="#bdc8ca"
NAMES={"jepa_joint":"JCPR joint","graph_autoencoder_joint":"Graph autoencoder",
 "pca_graph_statistics":"Graph-statistics PCA","input_graph_statistics":"Input graph statistics",
 "contrastive_joint":"Contrastive graph encoder","random_joint":"Untrained graph encoder",
 "brepnet_official_histcad":"BRepNet transfer"}

def read(path):
    with path.open(newline="",encoding="utf-8") as f:
        return {r["representation"]:r for r in csv.DictReader(f)}

def output(fig,name):
    fig.savefig(HERE/f"{name}.pdf",bbox_inches="tight",pad_inches=.045)
    preview=HERE/"qa_previews";preview.mkdir(exist_ok=True)
    fig.savefig(preview/f"{name}.png",dpi=210,bbox_inches="tight",pad_inches=.045)
    plt.close(fig)

def label(ax,letter):
    ax.text(-.055,1.025,letter,transform=ax.transAxes,fontsize=9.5,
            fontweight="bold",ha="right",va="bottom")

def paired(ax,names,first,second,xmax,ticks,xlabel):
    y=np.arange(len(names))[::-1]
    for yi,n in zip(y,names):
        ax.plot([first[n],second[n]],[yi,yi],color=GRAY,lw=1.25,zorder=1)
    ax.scatter([first[n] for n in names],y,s=27,color=BLUE,label="Direct vector",zorder=3)
    ax.scatter([second[n] for n in names],y,s=27,color=ORANGE,label="Numeric adapter",zorder=3)
    ax.set_yticks(y,[NAMES[n] for n in names]);ax.set_ylim(-.75,len(names)-.25)
    ax.set_xlim(0,xmax);ax.set_xticks(ticks);ax.set_xlabel(xlabel)
    ax.grid(axis="x",color="#e6eaee",lw=.55);ax.set_axisbelow(True)
    ax.tick_params(axis="y",length=0);ax.minorticks_off()

def numeric_benchmark():
    data=read(ROOT/"benchmarking/outputs/retrieval_adaptation/summary.csv")
    names=["jepa_joint","graph_autoencoder_joint","contrastive_joint",
           "pca_graph_statistics","random_joint","brepnet_official_histcad"]
    fig,axes=plt.subplots(2,1,figsize=(7.15,4.92))
    fig.subplots_adjust(left=.305,right=.985,top=.87,bottom=.11,hspace=.46)
    specs=[("direct_dimension_recall_at_10","adapted_dimension_recall_at_10",
            .64,np.arange(0,.7,.1),"Dimensional neighbor recall@10"),
           ("direct_dimension_mae_at_1","adapted_dimension_mae_at_1",
            .60,np.arange(0,.7,.1),"Dimensional MAE@1 (standardized units)")]
    for ax,letter,(before,after,xmax,ticks,xlabel) in zip(axes,"ab",specs):
        paired(ax,names,{n:float(data[n][before]) for n in names},
               {n:float(data[n][after]) for n in names},xmax,ticks,xlabel)
        label(ax,letter)
        for yi,name in zip(np.arange(len(names))[::-1],names):
            left,right=sorted((float(data[name][before]),
                               float(data[name][after])))
            ax.annotate(f"{left:.3f}",(left,yi),(-6,0),
                        textcoords="offset points",ha="right",va="center",
                        fontsize=6.8,color="black")
            ax.annotate(f"{right:.3f}",(right,yi),(6,0),
                        textcoords="offset points",ha="left",va="center",
                        fontsize=6.8,color="black")
    handles,legends=axes[0].get_legend_handles_labels()
    fig.legend(handles,legends,ncol=2,frameon=False,loc="upper right",bbox_to_anchor=(.985,.995))
    output(fig,"histcad_numeric_benchmark")

def semantic_tradeoff():
    data=read(ROOT/"benchmarking/outputs/semantic_retrieval/summary.csv")
    names=["jepa_joint","graph_autoencoder_joint","pca_graph_statistics",
           "input_graph_statistics","contrastive_joint","random_joint",
           "brepnet_official_histcad"]
    fig,axes=plt.subplots(2,1,figsize=(7.15,5.0))
    fig.subplots_adjust(left=.305,right=.96,top=.88,bottom=.10,hspace=.46)
    specs=[("direct_semantic_hit_at_1","adapted_semantic_hit_at_1",
            .47,[0,.1,.2,.3,.4],"Canonical-function match@1"),
           ("direct_dimension_recall_at_10","adapted_dimension_recall_at_10",
            .34,[0,.1,.2,.3],"Dimensional neighbor recall@10")]
    for ax,letter,(before,after,xmax,ticks,xlabel) in zip(axes,"ab",specs):
        y=np.arange(len(names))[::-1];h=.40
        ax.barh(y+h/2,[float(data[n][before]) for n in names],h,color=BLUE,label="Direct vector")
        ax.barh(y-h/2,[float(data[n][after]) for n in names],h,color=ORANGE,label="Text adapter")
        ax.set_yticks(y,[NAMES[n] for n in names]);ax.set_ylim(-.65,len(names)-.35)
        ax.set_xlim(0,xmax);ax.set_xticks(ticks);ax.set_xlabel(xlabel)
        ax.grid(axis="x",color="#e6eaee",lw=.55);ax.set_axisbelow(True)
        ax.tick_params(axis="y",length=0);label(ax,letter)
        for yi,name in zip(y,names):
            for key,position in ((before,yi+h/2),(after,yi-h/2)):
                value=float(data[name][key])
                ax.text(value+.004,position,f"{value:.3f}",ha="left",va="center",
                        fontsize=7.0,color="black")
    handles,legends=axes[0].get_legend_handles_labels()
    fig.legend(handles,legends,ncol=2,frameon=False,loc="upper right",bbox_to_anchor=(.985,.995))
    output(fig,"semantic_tradeoff")

def confusion_change():
    path=ROOT/"downstream_classifier_tmcad_final/outputs/default/test_predictions.csv"
    with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    selected=["coupling","pulley","gear","flange"]
    labels=["Coupling","Pulley","Gear","Flange","Other"]
    groups=[("coupling","baseline_prediction"),("coupling","final_prediction"),
            ("pulley","baseline_prediction"),("pulley","final_prediction")]
    counts=np.zeros((4,5),dtype=int)
    for i,(truth,key) in enumerate(groups):
        own=[r for r in rows if r["true_class"]==truth]
        for r in own:
            prediction=r[key]
            counts[i,selected.index(prediction) if prediction in selected else 4]+=1
    ratios=counts.astype(float)/np.array([70,70,78,78])[:,None]*100
    classes=sorted({r["true_class"] for r in rows})
    totals={c:sum(r["true_class"]==c for r in rows) for c in classes}
    recall_control={c:100*sum(r["true_class"]==c and r["baseline_prediction"]==c for r in rows)/totals[c]
                    for c in classes}
    recall_residual={c:100*sum(r["true_class"]==c and r["final_prediction"]==c for r in rows)/totals[c]
                     for c in classes}
    recall_change={c:recall_residual[c]-recall_control[c] for c in classes}
    ordered=sorted(classes,key=lambda c:(-recall_change[c],c))
    fig=plt.figure(figsize=(7.15,3.95))
    a=fig.add_axes([.14,.18,.47,.75])
    ax=fig.add_axes([.71,.18,.28,.75])
    yy=np.arange(len(ordered))[::-1]
    for y,c in zip(yy,ordered):
        a.plot([recall_control[c],recall_residual[c]],[y,y],color=GRAY,lw=1.1,zorder=1)
    a.scatter([recall_control[c] for c in ordered],yy,s=25,color=BLUE,
              label="STEP control",zorder=3)
    a.scatter([recall_residual[c] for c in ordered],yy,s=25,color=ORANGE,
              label="+ physical residual",zorder=3)
    a.set_yticks(yy,["Bolt/screw" if c=="bolt_screw" else c.capitalize() for c in ordered])
    a.set_xlim(68,101);a.set_xticks([70,80,90,100])
    a.set_xlabel("Class recall (%)")
    a.grid(axis="x",color="#e2e7e7",lw=.55);a.set_axisbelow(True)
    a.tick_params(axis="y",length=0)
    for y,c in zip(yy,ordered):
        for value,color,offset in ((recall_control[c],BLUE,7),(recall_residual[c],ORANGE,-8)):
            a.annotate(f"{value:.1f}",(value,y),(0,offset),textcoords="offset points",
                       ha="center",fontsize=6.7,color=color)
    a.legend(frameon=False,loc="lower left",fontsize=7.1,handletextpad=.2,
             borderaxespad=.1,ncol=1)
    label(a,"a")
    color_map=LinearSegmentedColormap.from_list("muted_blue",["#f5f7f7","#456778"])
    for i in range(4):
        for j in range(5):
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,
                facecolor=color_map(ratios[i,j]/100),edgecolor="white",lw=1.5))
    ax.set_xlim(-.5,4.5);ax.set_ylim(3.5,-.5)
    ax.set_xticks(range(5),labels,rotation=35,ha="right")
    ax.set_yticks(range(4),["Coupling\ncontrol","Coupling\nresidual",
                             "Pulley\ncontrol","Pulley\nresidual"])
    ax.tick_params(axis="y",labelsize=7.7)
    ax.set_xlabel("Predicted class");ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    for i in range(4):
        for j in range(5):
            value=ratios[i,j]
            ax.text(j,i,f"{value:.1f}%",ha="center",va="center",fontsize=8.5,
                    fontweight="bold" if (i,j) in ((1,0),(3,1)) else "normal",
                    color="white" if value>=60 else "#283f4b")
    label(ax,"b")
    output(fig,"tmcad_confusion_change")

def interpretability():
    audit=json.loads((HERE/"residual_audit.json").read_text(encoding="utf-8"))
    scores=audit["metrics"]
    names=["Full residual","Geometry masked",
           "Surface profile masked","Both groups masked"]
    keys=["full","geometry_masked","surface_masked","both_masked"]
    values=[scores[k]["macro_f1"] for k in keys]
    fig,(a,b)=plt.subplots(1,2,figsize=(7.15,3.05))
    fig.subplots_adjust(left=.225,right=.975,top=.91,bottom=.20,wspace=.68)
    y=np.arange(4)[::-1]
    a.hlines(y,.84,values,color="#c5d0d0",lw=1.15)
    a.scatter(values,y,s=42,color=[ORANGE,BLUE,BLUE,"#9daeb2"],zorder=3)
    a.set_yticks(y,names);a.set_xlim(.83,.94);a.set_xticks([.84,.88,.92]);a.set_xlabel("Macro-F1")
    a.grid(axis="x",color="#e6eaee",lw=.55);a.set_axisbelow(True);a.tick_params(axis="y",length=0)
    for yi,v in zip(y,values):a.text(v+.006,yi,f"{v:.3f}",va="center",fontsize=7.8)
    label(a,"a")
    effects=audit["group_margin_effects"]
    order=[("geometry","correct"),("geometry","incorrect"),
           ("surface_profile","correct"),("surface_profile","incorrect")]
    yy=np.arange(4)[::-1]
    for yi,(group,outcome) in zip(yy,order):
        row=effects[group][outcome]
        color=BLUE if group=="geometry" else ORANGE
        b.plot([row["q25"],row["q75"]],[yi,yi],lw=5.5,color=color,
               solid_capstyle="butt",alpha=.58)
        b.scatter(row["median"],yi,s=37,color=color,zorder=3)
        b.text(row["q75"]+.09,yi,f'{row["median"]:+.2f}',va="center",fontsize=7.4)
    b.set_yticks(yy,["Geometry · correct","Geometry · incorrect",
                     "Surface · correct","Surface · incorrect"])
    b.set_xlim(-.65,1.3);b.set_xticks([-.5,0,.5,1.0]);b.set_xlabel("True-class margin effect (logits)")
    b.axvline(0,color="#70838a",lw=.7);b.grid(axis="x",color="#e6eaee",lw=.55)
    b.set_axisbelow(True);b.tick_params(axis="y",length=0);label(b,"b")
    output(fig,"tmcad_residual_interpretability")

if __name__=="__main__":
    numeric_benchmark();semantic_tradeoff();confusion_change();interpretability()
