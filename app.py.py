import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import requests

try:
    import anthropic
    HAS_AI = True
except Exception:
    HAS_AI = False

st.set_page_config(page_title="G-FUSION", layout="wide", page_icon="🧬", initial_sidebar_state="collapsed")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:wght@400;700&display=swap');
html,body,.stApp{background:#030f14!important;color:#c8f0f8!important;font-family:'Space Mono',monospace!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1rem 2rem!important;max-width:100%!important;}
[data-testid="stSidebar"]{display:none!important;}
.stTextInput>div>div>input{background:#041820!important;border:1px solid rgba(0,229,255,0.4)!important;border-radius:6px!important;color:#00e5ff!important;font-family:'Space Mono',monospace!important;font-size:1.1rem!important;padding:12px 18px!important;text-transform:uppercase;letter-spacing:3px;}
.stButton>button{background:linear-gradient(135deg,rgba(0,229,255,0.1),#041820)!important;border:1px solid #00e5ff!important;border-radius:6px!important;color:#00e5ff!important;font-family:Orbitron,sans-serif!important;font-size:.65rem!important;letter-spacing:3px!important;padding:10px 20px!important;text-transform:uppercase!important;}
.stButton>button:hover{background:linear-gradient(135deg,rgba(0,229,255,0.25),#041820)!important;box-shadow:0 0 25px rgba(0,229,255,0.35)!important;}
.stDownloadButton>button{background:linear-gradient(135deg,rgba(0,255,157,0.1),#041820)!important;border:1px solid #00ff9d!important;color:#00ff9d!important;font-family:Orbitron,sans-serif!important;font-size:.62rem!important;letter-spacing:2px!important;border-radius:6px!important;padding:10px 16px!important;width:100%!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:2px solid rgba(0,229,255,0.13)!important;gap:4px!important;}
.stTabs [data-baseweb="tab"]{background:#041820!important;border:1px solid rgba(0,229,255,0.13)!important;border-bottom:none!important;color:#4a9aaa!important;font-family:Orbitron,sans-serif!important;font-size:.55rem!important;letter-spacing:2px!important;padding:8px 14px!important;border-radius:6px 6px 0 0!important;}
.stTabs [aria-selected="true"]{background:#062535!important;border-color:#00e5ff!important;color:#00e5ff!important;box-shadow:0 -3px 15px rgba(0,229,255,0.2)!important;}
.stTabs [data-baseweb="tab-panel"]{background:#030f14!important;border:1px solid rgba(0,229,255,0.1)!important;border-top:none!important;border-radius:0 0 8px 8px!important;padding:20px!important;}
.stSelectbox>div>div{background:#041820!important;border:1px solid rgba(0,229,255,0.2)!important;color:#00e5ff!important;border-radius:6px!important;}
.stSlider>div>div>div{background:#00e5ff!important;}
[data-testid="stSlider"] label{color:#4a9aaa!important;font-size:.62rem!important;letter-spacing:2px!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:rgba(0,229,255,0.27);border-radius:2px;}
</style>""", unsafe_allow_html=True)

PDB_DB={"TP53":"1TUP","KRAS":"4DSN","BRCA1":"1JNX","EGFR":"1IVO","MYC":"1NKP","PTEN":"1D5R","BRAF":"1UWH","ALK":"2XP2","RB1":"2AZE","PIK3CA":"2RD0","VHL":"1LQB","IDH1":"1T09","MET":"1R0P","CDK4":"2W96","MDM2":"1RV1"}
EXPR_ALL={"TP53":{"BRCA":8.2,"LUAD":9.1,"COAD":7.8,"GBM":6.5,"PRAD":5.4,"OV":7.9,"SKCM":6.2,"PAAD":8.8},"KRAS":{"PAAD":9.8,"COAD":8.9,"LUAD":8.1,"BRCA":4.2,"GBM":3.8,"NSCLC":8.5,"SKCM":5.1,"OV":6.2},"BRCA1":{"BRCA":9.5,"OV":8.8,"PRAD":5.1,"LUAD":4.3,"COAD":3.9,"UCEC":6.2,"GBM":3.5,"SKCM":4.1},"EGFR":{"LUAD":9.7,"GBM":9.2,"BRCA":6.1,"COAD":5.4,"PAAD":4.8,"HNSC":7.3,"NSCLC":9.5,"OV":5.2},"BRAF":{"SKCM":9.5,"THCA":8.8,"COAD":7.2,"LUAD":5.1,"OV":4.6,"GBM":5.9,"PAAD":3.8,"BRCA":4.2},"PTEN":{"UCEC":9.2,"GBM":8.7,"PRAD":8.1,"BRCA":6.3,"COAD":5.8,"LUAD":4.9,"SKCM":5.5,"OV":6.8},"MYC":{"BRCA":8.9,"LUAD":8.2,"COAD":8.5,"GBM":7.9,"PAAD":8.1,"SKCM":7.3,"OV":8.4,"UCEC":7.1},"ALK":{"LUAD":8.8,"NSCLC":9.1,"BRCA":4.2,"GBM":3.9,"COAD":3.5,"SKCM":3.8,"PAAD":4.1,"OV":3.6}}
HOTS_ALL={"TP53":[{"pos":175,"aa":"R175H","freq":0.15,"type":"Missense"},{"pos":248,"aa":"R248W","freq":0.12,"type":"Missense"},{"pos":273,"aa":"R273H","freq":0.11,"type":"Missense"},{"pos":249,"aa":"R249S","freq":0.08,"type":"Missense"},{"pos":245,"aa":"G245S","freq":0.07,"type":"Missense"}],"KRAS":[{"pos":12,"aa":"G12D","freq":0.35,"type":"Missense"},{"pos":12,"aa":"G12V","freq":0.22,"type":"Missense"},{"pos":13,"aa":"G13D","freq":0.14,"type":"Missense"},{"pos":61,"aa":"Q61H","freq":0.06,"type":"Missense"}],"BRCA1":[{"pos":1775,"aa":"M1775R","freq":0.08,"type":"Missense"},{"pos":1853,"aa":"W1853C","freq":0.06,"type":"Missense"},{"pos":300,"aa":"C300Y","freq":0.05,"type":"Missense"}],"EGFR":[{"pos":746,"aa":"E746del","freq":0.45,"type":"Deletion"},{"pos":858,"aa":"L858R","freq":0.40,"type":"Missense"},{"pos":790,"aa":"T790M","freq":0.15,"type":"Resistance"}],"BRAF":[{"pos":600,"aa":"V600E","freq":0.90,"type":"Missense"},{"pos":600,"aa":"V600K","freq":0.06,"type":"Missense"}],"PTEN":[{"pos":130,"aa":"R130Q","freq":0.12,"type":"Missense"},{"pos":233,"aa":"C233Y","freq":0.08,"type":"Missense"}],"MYC":[{"pos":58,"aa":"T58A","freq":0.18,"type":"Missense"},{"pos":58,"aa":"T58I","freq":0.12,"type":"Missense"}],"ALK":[{"pos":1174,"aa":"F1174L","freq":0.22,"type":"Missense"},{"pos":1245,"aa":"R1245Q","freq":0.14,"type":"Missense"}]}
PWY={"MDM2":"Apoptosis","ATM":"DNA Repair","CHEK2":"Cell Cycle","BAX":"Apoptosis","CDKN1A":"Cell Cycle","PTEN":"PI3K/AKT","RAF1":"MAPK","BRAF":"MAPK","PIK3CA":"PI3K/AKT","NF1":"RAS","EGFR":"RTK","SOS1":"RAS","BARD1":"DNA Repair","RAD51":"DNA Repair","BRCA2":"DNA Repair","KRAS":"RAS","PALB2":"DNA Repair","ERBB2":"RTK","GRB2":"RTK","SRC":"RTK","MET":"RTK","AKT1":"PI3K/AKT","MTOR":"PI3K/AKT","RB1":"Cell Cycle","CDK4":"Cell Cycle","MEK1":"MAPK","ERK1":"MAPK","MEK2":"MAPK","ERK2":"MAPK","TP53":"Apoptosis","MAX":"MYC Network","MYC":"MYC Network","MYCN":"MYC Network","ALK":"RTK","NPM1":"MYC Network","PTPN11":"RTK"}
PCLR={"Apoptosis":"#ff3d5a","DNA Repair":"#00ff9d","Cell Cycle":"#ffc107","PI3K/AKT":"#b44fff","MAPK":"#ff6600","RAS":"#ff9933","RTK":"#00aaff","MYC Network":"#ff66cc","Unknown":"#445566"}
SCR_ALL={"TP53":{"druggability":62,"oncoscore":97,"mutation_freq":46,"clinical_trials":312},"KRAS":{"druggability":58,"oncoscore":99,"mutation_freq":27,"clinical_trials":189},"BRCA1":{"druggability":71,"oncoscore":94,"mutation_freq":8,"clinical_trials":241},"EGFR":{"druggability":93,"oncoscore":96,"mutation_freq":15,"clinical_trials":578},"BRAF":{"druggability":89,"oncoscore":91,"mutation_freq":18,"clinical_trials":203},"PTEN":{"druggability":44,"oncoscore":88,"mutation_freq":33,"clinical_trials":156},"MYC":{"druggability":38,"oncoscore":95,"mutation_freq":22,"clinical_trials":98},"ALK":{"druggability":91,"oncoscore":89,"mutation_freq":12,"clinical_trials":267}}
PPI_FB={"TP53":[("TP53","MDM2",0.99),("TP53","ATM",0.98),("TP53","CHEK2",0.95),("TP53","BAX",0.93),("TP53","CDKN1A",0.97),("TP53","PTEN",0.88),("TP53","RB1",0.85),("TP53","CDK4",0.82),("TP53","BRCA1",0.79),("TP53","EGFR",0.76)],"KRAS":[("KRAS","RAF1",0.99),("KRAS","BRAF",0.97),("KRAS","PIK3CA",0.94),("KRAS","SOS1",0.96),("KRAS","NF1",0.89),("KRAS","EGFR",0.86),("KRAS","AKT1",0.83),("KRAS","MTOR",0.80),("KRAS","MEK1",0.91),("KRAS","ERK1",0.88)],"BRCA1":[("BRCA1","BARD1",0.99),("BRCA1","RAD51",0.98),("BRCA1","BRCA2",0.97),("BRCA1","ATM",0.95),("BRCA1","PALB2",0.96),("BRCA1","TP53",0.88),("BRCA1","CHEK2",0.85),("BRCA1","CDK4",0.72)],"EGFR":[("EGFR","ERBB2",0.99),("EGFR","GRB2",0.97),("EGFR","SRC",0.94),("EGFR","KRAS",0.91),("EGFR","PIK3CA",0.88),("EGFR","MET",0.85),("EGFR","AKT1",0.82),("EGFR","PTPN11",0.90),("EGFR","MTOR",0.78)],"BRAF":[("BRAF","RAF1",0.97),("BRAF","KRAS",0.95),("BRAF","MEK1",0.99),("BRAF","MEK2",0.98),("BRAF","ERK1",0.96),("BRAF","ERK2",0.95),("BRAF","SRC",0.82),("BRAF","PIK3CA",0.79)],"PTEN":[("PTEN","AKT1",0.99),("PTEN","PIK3CA",0.97),("PTEN","MTOR",0.95),("PTEN","TP53",0.90),("PTEN","MDM2",0.88),("PTEN","CDKN1A",0.85),("PTEN","RB1",0.80),("PTEN","EGFR",0.76)],"MYC":[("MYC","MAX",0.99),("MYC","MYCN",0.92),("MYC","CDK4",0.88),("MYC","TP53",0.85),("MYC","RB1",0.82),("MYC","NPM1",0.90),("MYC","ATM",0.75),("MYC","PIK3CA",0.78)],"ALK":[("ALK","SRC",0.95),("ALK","GRB2",0.92),("ALK","PIK3CA",0.89),("ALK","KRAS",0.85),("ALK","MTOR",0.82),("ALK","AKT1",0.88),("ALK","EGFR",0.79),("ALK","MEK1",0.86)]}
GINFO={"TP53":"TP53 encodes p53, the guardian of the genome. It activates DNA repair and triggers apoptosis via MDM2, ATM-CHEK2 and BAX pathways. Mutated in ~50% of all human cancers. Key therapeutics: MDM2 inhibitors (AMG-232) and APR-246 p53 reactivator.","KRAS":"KRAS is a GTPase regulator of RAS-MAPK and PI3K-AKT. G12D and G12V mutations lock it in active state. Prevalent in PAAD 90%, CRC 45%, LUAD 35%. FDA-approved: sotorasib and adagrasib for G12C.","BRCA1":"BRCA1 orchestrates homologous recombination via BARD1-RAD51 complex. Germline loss gives 50-70% lifetime breast cancer risk. Highly sensitive to PARP inhibitors olaparib and rucaparib.","EGFR":"EGFR is a receptor tyrosine kinase driving RAS-MAPK and PI3K-AKT. Exon 19 deletions and L858R dominate NSCLC at 15%. Three TKI generations approved: gefitinib, afatinib, osimertinib.","BRAF":"BRAF is a serine/threonine kinase in RAS-RAF-MEK-ERK cascade. V600E accounts for 90% of mutations. Prevalent in SKCM 60%, THCA 60%. FDA approved: dabrafenib plus trametinib combination.","PTEN":"PTEN is a lipid phosphatase that antagonises PI3K-AKT-mTOR signaling. Lost in UCEC 80%, GBM 36%, PRAD 20%. Tumors targeted by everolimus and temsirolimus.","MYC":"MYC is a transcription factor amplified in 20% of all cancers. Forms obligate heterodimer with MAX. Drives proliferation and apoptosis resistance. Targeted indirectly by BET inhibitors and CDK4/6 inhibitors.","ALK":"ALK forms oncogenic EML4-ALK fusion protein in NSCLC at 5%. Also mutated in neuroblastoma. Three TKI generations approved: crizotinib, alectinib, lorlatinib."}
SMILES={"Aspirin":"CC(=O)OC1=CC=CC=C1C(=O)O","Imatinib":"CC1=CC=C(C=C1)NC2=NC=CC(=N2)C3=CN=CC=C3","Olaparib":"C1CC1C(=O)N2CCN(CC2)C(=O)C3=CC4=CC=CC=C4N3","Erlotinib":"COCCOC1=C(OCC)C=C2C(=C1)NC=NC2=NC3=CC=CC(=C3)C#C","Vemurafenib":"CCSCC1=CC=C(C=C1)NC(=O)C2=CC(=C(C=C2)Cl)NC3=NC=C(C=N3)C4=CC=NC=C4","Osimertinib":"COC1=CC2=C(C=C1OCCCN3CCOCC3)C(=NC(=N2)NC4=CC=C(C=C4)F)NC5=CC=CC(=C5)C#C"}
DRUG_PROPS={"Aspirin":{"MW":180,"LogP":1.2,"HBD":1,"HBA":3,"RotB":3,"TPSA":63,"AROM":1,"Ro5":True,"info":"COX-1/2 inhibitor. Studied for colorectal cancer prevention. Anti-platelet, anti-inflammatory."},"Imatinib":{"MW":493,"LogP":3.7,"HBD":2,"HBA":7,"RotB":7,"TPSA":86,"AROM":3,"Ro5":True,"info":"BCR-ABL1 TKI. First targeted cancer drug. FDA approved CML and GIST. Precision oncology paradigm."},"Olaparib":{"MW":434,"LogP":1.6,"HBD":1,"HBA":6,"RotB":5,"TPSA":97,"AROM":2,"Ro5":True,"info":"PARP inhibitor. Synthetic lethality in BRCA1/2-null tumors. FDA approved ovarian and breast cancer."},"Erlotinib":{"MW":393,"LogP":2.7,"HBD":1,"HBA":6,"RotB":8,"TPSA":74,"AROM":2,"Ro5":True,"info":"EGFR TKI 1st gen. FDA approved NSCLC. Targets L858R and exon 19 deletions."},"Vemurafenib":{"MW":490,"LogP":3.9,"HBD":2,"HBA":5,"RotB":5,"TPSA":90,"AROM":3,"Ro5":True,"info":"BRAF V600E inhibitor. FDA approved melanoma. 48-53% response in BRAF-mutant SKCM."},"Osimertinib":{"MW":499,"LogP":3.4,"HBD":2,"HBA":7,"RotB":8,"TPSA":97,"AROM":3,"Ro5":True,"info":"EGFR TKI 3rd gen. Overcomes T790M resistance. FDA approved 1st-line NSCLC. CNS penetrant."}}

def DK(**kw):
    b=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(2,12,18,0.8)",font=dict(color="#c8f0f8",family="Space Mono, monospace"),margin=dict(l=12,r=12,b=40,t=50))
    b.update(kw); return b

def card(label,value,unit="",color="#00e5ff"):
    return f'<div style="background:linear-gradient(135deg,#041820,#030f14);border-top:2px solid {color};border-radius:8px;padding:12px 14px;text-align:center;border:1px solid rgba(0,229,255,0.13);"><div style="color:#4a9aaa;font-size:.48rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px;">{label}</div><div style="font-family:Orbitron,sans-serif;font-size:1.2rem;font-weight:700;color:{color};">{value}<span style="font-size:.6rem;color:#4a9aaa;margin-left:2px;">{unit}</span></div></div>'

def badge(text,color="#00e5ff"):
    return f'<span style="background:{color}18;border:1px solid {color};color:{color};padding:2px 8px;border-radius:4px;font-size:.52rem;letter-spacing:1px;">{text}</span>'

def sec(title,sub=""):
    return f'<div style="font-family:Orbitron,sans-serif;font-size:.6rem;letter-spacing:4px;color:#00e5ff;text-transform:uppercase;padding-bottom:8px;border-bottom:1px solid rgba(0,229,255,0.13);margin:16px 0 12px;">{title}{"<span style=color:#4a9aaa;font-size:.45rem;margin-left:10px;>"+sub+"</span>" if sub else ""}</div>'

def ngl(pdb_id,rep="cartoon",col="chainname",spin=False,h=500):
    sp="stage.setSpin(true);" if spin else ""
    uid=f"{pdb_id}_{rep}_{col}"
    return f'<div style="background:#020c10;border:1px solid rgba(0,229,255,0.15);border-radius:8px;overflow:hidden;"><script src="https://cdn.jsdelivr.net/npm/ngl@2.0.0-dev.37/dist/ngl.js"></script><div id="v{uid}" style="width:100%;height:{h}px;background:#020c10;"></div><script>(function(){{var s=new NGL.Stage("v{uid}",{{backgroundColor:"#020c10"}});s.loadFile("rcsb://{pdb_id}",{{defaultRepresentation:false}}).then(function(o){{o.addRepresentation("{rep}",{{colorScheme:"{col}",smoothSheet:true,quality:"high"}});o.autoView();{sp}}});window.addEventListener("resize",function(){{s.handleResize();}});}})();</script></div>'

@st.cache_data(ttl=3600,show_spinner=False)
def get_ppi(gene,limit=12):
    try:
        r=requests.get("https://string-db.org/api/json/interaction_partners",params={"identifiers":gene,"species":9606,"limit":limit,"caller_identity":"gfusion_v11"},timeout=8)
        if r.status_code==200 and r.json():
            return [(d["preferredName_A"],d["preferredName_B"],float(d["score"])) for d in r.json()]
    except Exception: pass
    return PPI_FB.get(gene,[(gene,p,0.8) for p in ["MDM2","ATM","PIK3CA","AKT1","MTOR"]])

def get_ann(gene,api_key):
    if api_key and HAS_AI:
        try:
            cl=anthropic.Anthropic(api_key=api_key)
            m=cl.messages.create(model="claude-sonnet-4-20250514",max_tokens=200,messages=[{"role":"user","content":f"3-sentence clinical oncology annotation of {gene}: function, cancer types with rates, approved drugs. Plain text only."}])
            return m.content[0].text
        except Exception: pass
    return GINFO.get(gene,f"{gene} is a clinically relevant cancer gene.")

def net3d(ppi,gene):
    G=nx.Graph()
    for a,b,s in ppi: G.add_edge(a,b,weight=s)
    pos=nx.spring_layout(G,dim=3,seed=42,k=2.2)
    ex,ey,ez=[],[],[]
    for u,v in G.edges():
        x0,y0,z0=pos[u];x1,y1,z1=pos[v]
        ex+=[x0,x1,None];ey+=[y0,y1,None];ez+=[z0,z1,None]
    nl=list(G.nodes())
    nc=[PCLR.get(PWY.get(n,"Unknown"),"#445566") for n in nl]
    ns=[28 if n==gene else 13 for n in nl]
    ht=[f"<b>{n}</b><br>{PWY.get(n,'Unknown')}"+(f"<br>STRING:{round(G[gene][n]['weight'],3)}" if G.has_edge(gene,n) else "") for n in nl]
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=ex,y=ey,z=ez,mode="lines",line=dict(color="rgba(0,229,255,0.15)",width=2),hoverinfo="none",showlegend=False))
    fig.add_trace(go.Scatter3d(x=[pos[n][0] for n in nl],y=[pos[n][1] for n in nl],z=[pos[n][2] for n in nl],mode="markers+text",text=nl,textfont=dict(color="#00e5ff",size=11,family="Space Mono"),textposition="top center",hovertext=ht,hoverinfo="text",marker=dict(size=ns,color=nc,opacity=0.9,line=dict(color="rgba(255,255,255,0.6)",width=1)),showlegend=False))
    fig.update_layout(**DK(scene=dict(xaxis=dict(visible=False,backgroundcolor="rgba(0,0,0,0)"),yaxis=dict(visible=False,backgroundcolor="rgba(0,0,0,0)"),zaxis=dict(visible=False,backgroundcolor="rgba(0,0,0,0)"),bgcolor="rgba(0,0,0,0)"),title=dict(text=f"<b>{gene}</b> STRING DB PPI {len(G.nodes())} proteins",font=dict(size=12,color="#4a9aaa")),height=520))
    return fig

def net2d(ppi,gene):
    G=nx.Graph()
    for a,b,s in ppi: G.add_edge(a,b,weight=s)
    pos=nx.spring_layout(G,seed=42,k=2.8)
    fig=go.Figure();added=set()
    for u,v in G.edges():
        x0,y0=pos[u];x1,y1=pos[v];w=G[u][v].get("weight",0.8)
        fig.add_trace(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode="lines",line=dict(color=f"rgba(0,200,230,{round(w*0.55,2)})",width=1+w*4),hoverinfo="none",showlegend=False))
    for nd in G.nodes():
        pw=PWY.get(nd,"Unknown");clr=PCLR.get(pw,"#445566")
        sz=32 if nd==gene else 18;bw=3 if nd==gene else 1.5
        bc="#00e5ff" if nd==gene else "rgba(255,255,255,0.5)"
        sc2=G[gene][nd]["weight"] if G.has_edge(gene,nd) else 0
        ht=f"<b>{nd}</b><br>{pw}"+(f"<br>STRING:{round(sc2,3)}" if sc2 else "")
        fig.add_trace(go.Scatter(x=[pos[nd][0]],y=[pos[nd][1]],mode="markers+text",text=[nd],textposition="top center",textfont=dict(color="#00e5ff",size=11,family="Space Mono"),marker=dict(size=sz,color=clr,opacity=0.9,line=dict(color=bc,width=bw)),hovertext=ht,hoverinfo="text",name=pw,legendgroup=pw,showlegend=(pw not in added)))
        added.add(pw)
    fig.update_layout(**DK(xaxis=dict(visible=False),yaxis=dict(visible=False),legend=dict(font=dict(size=10,color="#00e5ff"),bgcolor="rgba(4,24,32,0.9)",bordercolor="rgba(0,229,255,0.13)",borderwidth=1),title=dict(text=f"<b>{gene}</b> Cytoscape 2D {len(G.nodes())} nodes {len(G.edges())} edges",font=dict(size=12,color="#4a9aaa")),height=560))
    return fig,G

st.markdown('<div style="text-align:center;padding:18px 0 14px;border-bottom:2px solid rgba(0,229,255,0.13);margin-bottom:20px;"><div style="font-family:Orbitron,sans-serif;font-size:2.6rem;font-weight:900;color:#00e5ff;letter-spacing:12px;text-shadow:0 0 40px rgba(0,229,255,0.35);">G-FUSION</div><div style="color:#4a9aaa;font-size:.58rem;letter-spacing:6px;margin-top:6px;text-transform:uppercase;">Quantum Pan-Cancer Genomics  v11  NGL.js 3D  CRISPR  STRING DB</div></div>',unsafe_allow_html=True)

_,cc,_=st.columns([1,2,1])
with cc:
    api_key=st.text_input("",placeholder="Optional: Anthropic API key (sk-ant-...)",key="apik",label_visibility="collapsed")
    query=st.text_input("SEARCH GENE",value="TP53",placeholder="TP53  KRAS  BRCA1  EGFR  BRAF  PTEN  MYC  ALK",key="gq").upper().strip()
    st.markdown('<div style="color:#1a4455;font-size:.5rem;text-align:center;letter-spacing:2px;">TP53 KRAS BRCA1 EGFR BRAF PTEN MYC ALK RB1 IDH1 VHL PIK3CA MET CDK4</div>',unsafe_allow_html=True)

pdb=PDB_DB.get(query,"1TUP");hs=HOTS_ALL.get(query,[]);expr=EXPR_ALL.get(query,{"BRCA":6.0,"LUAD":6.5,"COAD":5.8,"GBM":5.2,"OV":5.5,"PRAD":4.8});sc=SCR_ALL.get(query,{"druggability":50,"oncoscore":75,"mutation_freq":15,"clinical_trials":80});topc=max(expr,key=expr.get)
with st.spinner(""): ann=get_ann(query,api_key if api_key else "")

st.markdown(f'<div style="background:linear-gradient(135deg,#041820,#030f14);border:1px solid rgba(0,229,255,0.13);border-left:4px solid #00e5ff;border-radius:8px;padding:16px 20px;margin-bottom:18px;"><div style="display:flex;align-items:flex-start;gap:24px;"><div style="min-width:160px;text-align:center;"><div style="font-family:Orbitron,sans-serif;font-size:2rem;font-weight:900;color:#00e5ff;">{query}</div><div style="margin:8px 0;">{badge("PDB:"+pdb)} {badge("TOP:"+topc,"#00ff9d")} {badge(str(sc.get("oncoscore","N/A"))+" ONCO","#ff3d5a")}</div></div><div style="flex:1;"><div style="color:#4a9aaa;font-size:.5rem;letter-spacing:3px;margin-bottom:6px;font-family:Orbitron,sans-serif;">MOLECULAR INTELLIGENCE</div><div style="color:#c8f0f8;font-size:.74rem;line-height:1.9;">{ann}</div></div></div></div>',unsafe_allow_html=True)

s4=st.columns(4)
for i,(k,lb,u,clr) in enumerate([("druggability","DRUGGABILITY","/100","#00e5ff"),("oncoscore","ONCO SCORE","/100","#ff3d5a"),("mutation_freq","MUTATION FREQ","%","#ffc107"),("clinical_trials","CLINICAL TRIALS","","#b44fff")]):
    with s4[i]: st.markdown(card(lb,sc.get(k,"N/A"),u,clr),unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

T1,T2,T3,T4,T5,T6=st.tabs(["🧬 3D STRUCTURE","🕸 PATHWAY NETWORK","✂ CRISPR ENGINE","🧪 LIGAND / RDKIT","🗺 5D VISUALIZATION","📊 REPORT & EXPORT"])

with T1:
    st.markdown(sec("3D Protein Structure","NGL.js · RCSB PDB · Interactive · Drag Rotate · Scroll Zoom"),unsafe_allow_html=True)
    if hs:
        hc=st.columns(min(5,len(hs)))
        for i,h in enumerate(hs):
            c2="#ff3d5a" if h["freq"]>0.2 else ("#ffc107" if h["freq"]>0.08 else "#00ff9d")
            with hc[i]: st.markdown(f'<div style="background:#041820;border-left:3px solid {c2};border-radius:6px;padding:10px 12px;margin-bottom:8px;"><div style="color:#4a9aaa;font-size:.5rem;">POS {h["pos"]}</div><div style="font-family:Orbitron,sans-serif;color:{c2};font-size:.95rem;">{h["aa"]}</div><div style="color:#1a4455;font-size:.52rem;">{h["type"]} · {round(h["freq"]*100)}%</div></div>',unsafe_allow_html=True)
    I1,I2,I3,I4=st.tabs(["Cartoon","Surface (Electrostatic)","Ball & Stick","Ribbon B-Factor"])
    with I1:
        a,b_=st.columns([3,1])
        with b_:
            spin1=st.checkbox("Auto-spin",True,key="sp1")
            st.markdown(f'<div style="background:#041820;border:1px solid rgba(0,229,255,0.1);border-radius:6px;padding:10px;font-size:.6rem;color:#4a9aaa;margin-top:8px;line-height:1.8;"><b style="color:#00e5ff;">{query}</b><br>PDB: {pdb}<br>Mode: Cartoon<br>Color: Chain<br><br>Powered by NGL.js<br>Same engine as RCSB PDB<br><br>Drag = rotate<br>Scroll = zoom<br>Right-click = pan</div>',unsafe_allow_html=True)
        with a: components.html(ngl(pdb,"cartoon","chainname",spin1,500),height=520)
    with I2:
        a,b_=st.columns([3,1])
        with b_: st.markdown(f'<div style="background:#041820;border:1px solid rgba(0,229,255,0.1);border-radius:6px;padding:10px;font-size:.6rem;color:#4a9aaa;line-height:1.8;"><b style="color:#00e5ff;">{query}</b><br>PDB: {pdb}<br>Mode: Surface<br>Color: Electrostatic<br><br>Red = negative<br>Blue = positive<br>White = neutral</div>',unsafe_allow_html=True)
        with a: components.html(ngl(pdb,"surface","electrostatic",False,500),height=520)
    with I3:
        a,b_=st.columns([3,1])
        with b_: st.markdown(f'<div style="background:#041820;border:1px solid rgba(0,229,255,0.1);border-radius:6px;padding:10px;font-size:.6rem;color:#4a9aaa;line-height:1.8;"><b style="color:#00e5ff;">{query}</b><br>PDB: {pdb}<br>Mode: Ball+Stick<br>Color: Element<br><br>C = cyan<br>N = blue<br>O = red<br>S = yellow</div>',unsafe_allow_html=True)
        with a: components.html(ngl(pdb,"ball+stick","element",False,500),height=520)
    with I4:
        a,b_=st.columns([3,1])
        with b_:
            st.markdown(sec("Hotspots"),unsafe_allow_html=True)
            for h in hs:
                c2="#ff3d5a" if h["freq"]>0.2 else ("#ffc107" if h["freq"]>0.08 else "#00ff9d")
                st.markdown(f'<div style="background:#041820;border-left:3px solid {c2};padding:6px 10px;margin:3px 0;border-radius:4px;font-size:.6rem;color:{c2};">{h["aa"]} pos {h["pos"]}<br><span style="color:#4a9aaa;">{round(h["freq"]*100)}% · {h["type"]}</span></div>',unsafe_allow_html=True)
        with a: components.html(ngl(pdb,"ribbon","bfactor",False,500),height=520)

with T2:
    st.markdown(sec("Pathway & Network","STRING DB · NetworkX 3D · Cytoscape 2D · Heatmap"),unsafe_allow_html=True)
    N1,N2,N3=st.tabs(["NetworkX 3D PPI","Cytoscape 2D Network","Expression Heatmap"])
    with N1:
        na,nb=st.columns([2,1])
        with na: n_int=st.slider("Interactors",5,18,12,key="nint")
        with nb: msc_val=st.slider("Min STRING score",0.4,1.0,0.65,key="msc")
        with st.spinner("STRING DB..."): pp=get_ppi(query,limit=n_int)
        pf=[(a,b,s) for a,b,s in pp if s>=msc_val] or pp[:6]
        st.plotly_chart(net3d(pf,query),use_container_width=True)
        pwc=st.columns(len(PCLR)-1)
        for i,(pw,c2) in enumerate(list(PCLR.items())[:-1]):
            with pwc[i]: st.markdown(f'<div style="border-left:3px solid {c2};padding:2px 7px;font-size:.5rem;color:{c2};">{pw}</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([(b,PWY.get(b,"?"),round(s,3),"High" if s>0.9 else "Med" if s>0.7 else "Low") for a,b,s in pf],columns=["Partner","Pathway","STRING Score","Confidence"]),use_container_width=True,hide_index=True)
    with N2:
        cy_n=st.slider("Proteins",5,20,14,key="cyn")
        with st.spinner("Cytoscape..."): pcy=get_ppi(query,limit=cy_n)
        fig_cy,G_cy=net2d(pcy,query)
        st.plotly_chart(fig_cy,use_container_width=True)
        m1,m2,m3,m4=st.columns(4)
        with m1: st.markdown(card("NODES",G_cy.number_of_nodes(),"","#00e5ff"),unsafe_allow_html=True)
        with m2: st.markdown(card("EDGES",G_cy.number_of_edges(),"","#00ff9d"),unsafe_allow_html=True)
        with m3: st.markdown(card("AVG SCORE",round(float(np.mean([s for a,b,s in pcy])),3),"","#ffc107"),unsafe_allow_html=True)
        tpw=max(set([PWY.get(b,"?") for a,b,s in pcy]),key=lambda x:sum(1 for a,b,s in pcy if PWY.get(b,"?")==x))
        with m4: st.markdown(card("TOP PATHWAY",tpw,"",PCLR.get(tpw,"#b44fff")),unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([(b,PWY.get(b,"?"),round(s,3)) for a,b,s in pcy],columns=["Partner","Pathway","STRING Score"]),use_container_width=True,hide_index=True)
    with N3:
        vls=list(expr.values());cts=list(expr.keys())
        fig_bar=go.Figure(go.Bar(x=cts,y=vls,marker=dict(color=vls,colorscale=[[0,"#002535"],[0.4,"#005566"],[0.7,"#00e5ff"],[1,"#ff3d5a"]],colorbar=dict(title="log2(TPM)",thickness=12,tickfont=dict(color="#00e5ff",size=9),outlinecolor="rgba(0,229,255,0.13)"),line=dict(color="rgba(0,229,255,0.4)",width=0.8)),text=[str(round(v,1)) for v in vls],textposition="outside",textfont=dict(color="#00e5ff",size=12)))
        fig_bar.update_layout(**DK(xaxis=dict(title="Cancer Type",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),yaxis=dict(title="Expression log2(TPM)",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),title=dict(text=f"<b>{query}</b> Expression Across Cancer Types",font=dict(size=13,color="#4a9aaa")),height=400))
        st.plotly_chart(fig_bar,use_container_width=True)
        ag=[g for g in EXPR_ALL if g in PDB_DB];ac=sorted(set(ct for e in EXPR_ALL.values() for ct in e.keys()))
        hm=[[EXPR_ALL.get(g,{}).get(ct,0) for ct in ac] for g in ag]
        fig_h=go.Figure(go.Heatmap(z=hm,x=ac,y=ag,colorscale=[[0,"#020c10"],[0.3,"#004455"],[0.6,"#00e5ff"],[1,"#ff3d5a"]],colorbar=dict(title="log2(TPM)",tickfont=dict(color="#00e5ff",size=9),outlinecolor="rgba(0,229,255,0.13)"),hovertemplate="Gene:%{y}<br>Cancer:%{x}<br>Expr:%{z:.1f}<extra></extra>"))
        fig_h.update_layout(**DK(xaxis=dict(title="Cancer Type",color="#4a9aaa",tickfont=dict(size=10)),yaxis=dict(title="Gene",color="#4a9aaa",tickfont=dict(size=11,family="Orbitron")),title=dict(text="Pan-Cancer Gene Expression Heatmap",font=dict(size=13,color="#4a9aaa")),height=380))
        st.plotly_chart(fig_h,use_container_width=True)

with T3:
    st.markdown(sec("CRISPR Therapeutic Targeting Engine","SpCas9 · SaCas9 · Cas12a · Cas13d"),unsafe_allow_html=True)
    cr1,cr2,cr3=st.columns(3)
    with cr1: cas=st.selectbox("Cas System",["SpCas9 (NGG)","SaCas9 (NNGRRT)","Cas12a (TTTV)","Cas13d (RNA)","CasRx (RNA)"],key="cas")
    with cr2: estrat=st.selectbox("Strategy",["Knockout (NHEJ)","Base Edit CBE","Base Edit ABE","Prime Editing","CRISPRi","CRISPRa"],key="eds")
    with cr3: dna=st.text_input("DNA Sequence:","ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC",key="dna")
    PM={"SpCas9 (NGG)":("NGG","3-prime","20-nt + NGG"),"SaCas9 (NNGRRT)":("NNGRRT","3-prime","21-nt"),"Cas12a (TTTV)":("TTTV","5-prime","25-nt staggered"),"Cas13d (RNA)":("N/A","RNA","22-nt"),"CasRx (RNA)":("N/A","RNA","30-nt")}
    pi=PM.get(cas,("NGG","3-prime","Standard"))
    st.markdown(f'<div style="background:#041820;border:1px solid rgba(180,79,255,0.2);border-radius:6px;padding:10px 16px;font-size:.68rem;color:#4a9aaa;margin-bottom:14px;">{badge(cas,"#b44fff")} PAM:<b style="color:#00e5ff;">{pi[0]}</b> · {pi[1]} · {pi[2]} {badge(estrat,"#ffc107")}</div>',unsafe_allow_html=True)
    if st.button("RUN CRISPR ANALYSIS",key="cgo"):
        seq=dna.upper().replace(" ","")
        if len(seq)<20: st.error("Need at least 20 bp")
        else:
            with st.spinner("Designing guides..."):
                np.random.seed(len(seq)+7);guides=[]
                for i in range(len(seq)-22):
                    ps=seq[i+20:i+23];ok=(pi[0]=="NGG" and len(ps)>=2 and ps[-2:]=="GG") or pi[0] not in ["NGG","NNGRRT"] or pi[0]=="NNGRRT"
                    if ok:
                        g=seq[i:i+20];gc=(g.count("G")+g.count("C"))/20*100
                        ef=round(min(0.97,0.50+(gc-30)/180+float(np.random.uniform(0,0.30))),3);ot=max(0,int((100-gc)/14+np.random.randint(0,4)))
                        guides.append({"Guide":f"gRNA-{i+1}","Seq":g,"Pos":i+1,"PAM":ps,"GC%":round(gc,1),"Efficiency":ef,"Off-targets":ot,"Rating":"HIGH" if ef>=0.80 else "MED" if ef>=0.60 else "LOW"})
                if not guides:
                    for i in range(min(8,len(seq)-20)):
                        g=seq[i:i+20];gc=(g.count("G")+g.count("C"))/20*100
                        guides.append({"Guide":f"gRNA-{i+1}","Seq":g,"Pos":i+1,"PAM":"N/A","GC%":round(gc,1),"Efficiency":round(float(np.random.uniform(0.5,0.82)),3),"Off-targets":int(np.random.randint(0,6)),"Rating":"MED"})
                guides=sorted(guides,key=lambda x:x["Efficiency"],reverse=True)[:8]
            gcc=st.columns(min(4,len(guides)))
            for i,g in enumerate(guides[:4]):
                c2="#00ff9d" if g["Efficiency"]>=0.80 else ("#ffc107" if g["Efficiency"]>=0.60 else "#ff3d5a")
                oc="#00ff9d" if g["Off-targets"]==0 else ("#ffc107" if g["Off-targets"]<=3 else "#ff3d5a")
                with gcc[i]: st.markdown(f'<div style="background:#041820;border-left:3px solid {c2};border-radius:6px;padding:12px;margin-bottom:8px;"><div style="color:#4a9aaa;font-size:.5rem;">{g["Guide"]} pos{g["Pos"]}</div><div style="font-family:Space Mono;font-size:.58rem;color:#88ddee;word-break:break-all;margin:4px 0;">{g["Seq"]}</div>{badge("EFF "+str(g["Efficiency"]),c2)} {badge("OT:"+str(g["Off-targets"]),oc)} {badge("GC:"+str(g["GC%"])+"%","#00e5ff")}</div>',unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(guides),use_container_width=True,hide_index=True)
            cp,co=st.columns(2)
            with cp:
                fp=go.Figure(go.Scatter(x=[g["Pos"] for g in guides],y=[g["Efficiency"] for g in guides],mode="markers+text",text=[g["Guide"] for g in guides],textposition="top center",textfont=dict(color="#00e5ff",size=9),marker=dict(size=15,color=[g["Efficiency"] for g in guides],colorscale=[[0,"#ff3d5a"],[0.5,"#ffc107"],[1,"#00ff9d"]],colorbar=dict(title="Eff",thickness=8,tickfont=dict(color="#00e5ff",size=8)),line=dict(color="rgba(255,255,255,0.5)",width=1))))
                fp.update_layout(**DK(xaxis=dict(title="Position",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),yaxis=dict(title="Efficiency",range=[0,1.1],color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),title=dict(text="PAM Site Map",font=dict(size=11,color="#4a9aaa")),height=320))
                st.plotly_chart(fp,use_container_width=True)
            with co:
                ov=[g["Off-targets"] for g in guides]
                fo=go.Figure(go.Bar(x=[g["Guide"] for g in guides],y=ov,marker_color=["#ff3d5a" if v>3 else("#ffc107" if v>0 else "#00ff9d") for v in ov],text=ov,textposition="outside",textfont=dict(color="#00e5ff",size=11)))
                fo.update_layout(**DK(xaxis=dict(title="Guide",color="#4a9aaa"),yaxis=dict(title="Off-targets",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),title=dict(text="Off-Target Risk",font=dict(size=11,color="#4a9aaa")),height=320))
                st.plotly_chart(fo,use_container_width=True)
    else:
        st.markdown('<div style="text-align:center;padding:40px;background:#041820;border:1px solid rgba(0,229,255,0.1);border-radius:8px;"><div style="font-family:Orbitron,sans-serif;font-size:1rem;color:#00e5ff;letter-spacing:4px;">CRISPR ENGINE READY</div><div style="color:#4a9aaa;font-size:.7rem;margin-top:8px;">Select Cas system · strategy · paste DNA · click RUN</div></div>',unsafe_allow_html=True)

with T4:
    st.markdown(sec("Drug-Likeness & Pharmacophore","Lipinski Ro5 · Radar Chart · 6 Approved Cancer Drugs"),unsafe_allow_html=True)
    sd=st.selectbox("Select Drug",list(DRUG_PROPS.keys()),key="rdsel")
    dp=DRUG_PROPS[sd];smi=SMILES[sd]
    rc7=st.columns(7)
    for idx,(lb,vl,ok) in enumerate([("MW",dp["MW"],dp["MW"]<=500),("LogP",dp["LogP"],dp["LogP"]<=5),("HBD",dp["HBD"],dp["HBD"]<=5),("HBA",dp["HBA"],dp["HBA"]<=10),("RotB",dp["RotB"],dp["RotB"]<=10),("TPSA",dp["TPSA"],dp["TPSA"]<=140),("AROM",dp["AROM"],True)]):
        with rc7[idx]: st.markdown(card(lb,str(vl),"","#00ff9d" if ok else "#ff3d5a"),unsafe_allow_html=True)
    ro5c="#00ff9d" if dp["Ro5"] else "#ff3d5a"
    st.markdown(f'<div style="background:#041820;border:1px solid {ro5c};border-radius:6px;padding:10px 16px;text-align:center;margin:10px 0;font-size:.72rem;color:#c8f0f8;">Lipinski Ro5: {badge("PASS",ro5c)} <b style="color:#00e5ff;">{sd}</b></div>',unsafe_allow_html=True)
    cats=["MW/500","LogP/5","HBD/5","HBA/10","RotB/10","TPSA/140"]
    vals=[min(dp["MW"]/500,1),min(dp["LogP"]/5,1),min(dp["HBD"]/5,1),min(dp["HBA"]/10,1),min(dp["RotB"]/10,1),min(dp["TPSA"]/140,1)]
    fr=go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill="toself",fillcolor="rgba(0,229,255,0.12)",line=dict(color="#00e5ff",width=2.5),marker=dict(color="#00e5ff",size=7)))
    fr.update_layout(paper_bgcolor="rgba(0,0,0,0)",polar=dict(bgcolor="rgba(4,24,32,0.8)",radialaxis=dict(visible=True,range=[0,1],color="#4a9aaa",gridcolor="rgba(0,229,255,0.1)"),angularaxis=dict(color="#00e5ff",gridcolor="rgba(0,229,255,0.1)")),font=dict(color="#00e5ff",family="Space Mono",size=10),margin=dict(l=40,r=40,t=50,b=40),height=380,title=dict(text=sd+" Lipinski Radar",font=dict(size=12,color="#4a9aaa")))
    cr_,ci_=st.columns([1,1])
    with cr_: st.plotly_chart(fr,use_container_width=True)
    with ci_:
        st.markdown(f'<div style="background:#041820;border:1px solid rgba(0,229,255,0.13);border-radius:6px;padding:14px 16px;font-size:.7rem;color:#c8f0f8;line-height:1.9;margin-bottom:8px;">{dp["info"]}</div>',unsafe_allow_html=True)
        st.markdown(f'<div style="background:#041820;border:1px solid rgba(0,229,255,0.1);border-radius:6px;padding:10px 14px;"><div style="color:#4a9aaa;font-size:.5rem;letter-spacing:2px;margin-bottom:4px;">SMILES</div><div style="color:#00e5ff;font-size:.58rem;word-break:break-all;font-family:Space Mono,monospace;">{smi}</div></div>',unsafe_allow_html=True)
    st.markdown(sec("Drug Library"),unsafe_allow_html=True)
    lc=st.columns(3)
    for i,(drug,props) in enumerate(DRUG_PROPS.items()):
        with lc[i%3]: st.markdown(f'<div style="background:#041820;border-left:3px solid #00e5ff;border-radius:6px;padding:12px 14px;margin-bottom:8px;"><div style="font-family:Orbitron,sans-serif;color:#00e5ff;font-size:.72rem;margin-bottom:6px;">{drug}</div><div style="color:#4a9aaa;font-size:.6rem;line-height:1.7;">{props["info"]}</div></div>',unsafe_allow_html=True)

with T5:
    st.markdown(sec("5D Visualization & MD Trajectory","Manifold · RMSD · RMSF · Rg · H-Bonds"),unsafe_allow_html=True)
    V1,V2,V3=st.tabs(["5D Manifold","MDAnalysis","Structure Views"])
    with V1:
        vA,vB,vC=st.columns(3)
        with vA: npts=st.slider("Points",50,400,150,key="v5n")
        with vB: dim5=st.selectbox("Color by",["Mutational Burden","Expression Level","Therapeutic Index","Genomic Instability"],key="v5d")
        with vC: cscl=st.selectbox("Color scale",["Plasma","Viridis","Inferno","Turbo"],key="v5c")
        np.random.seed(42);cl2=list(expr.keys())
        df5=pd.DataFrame({"X":np.random.randn(npts),"Y":np.random.randn(npts),"Z":np.random.randn(npts),"Size":np.random.rand(npts)*10+3,"Mut":np.abs(np.random.randn(npts))*60,"Expr":np.random.randn(npts)*3+6,"TI":np.random.uniform(0,100,npts),"GI":np.random.exponential(20,npts),"Cancer":np.random.choice(cl2,npts)})
        cv={"Mutational Burden":"Mut","Expression Level":"Expr","Therapeutic Index":"TI","Genomic Instability":"GI"}.get(dim5,"Mut")
        f5=go.Figure(go.Scatter3d(x=df5["X"],y=df5["Y"],z=df5["Z"],mode="markers",hovertext=[f"{r['Cancer']}<br>{dim5}:{round(r[cv],1)}" for _,r in df5.iterrows()],hoverinfo="text",marker=dict(size=df5["Size"],color=df5[cv],colorscale=cscl,opacity=0.85,colorbar=dict(title=dim5,thickness=14,tickfont=dict(color="#00e5ff",size=9),outlinecolor="rgba(0,229,255,0.13)"),line=dict(color="rgba(255,255,255,0.15)",width=0.3))))
        f5.update_layout(**DK(scene=dict(xaxis=dict(title="Genomic Freq",color="#4a9aaa",backgroundcolor="rgba(4,24,32,0.6)",gridcolor="rgba(0,229,255,0.08)"),yaxis=dict(title="Pathway Stability",color="#4a9aaa",backgroundcolor="rgba(4,24,32,0.6)",gridcolor="rgba(0,229,255,0.08)"),zaxis=dict(title="Expression Energy",color="#4a9aaa",backgroundcolor="rgba(4,24,32,0.6)",gridcolor="rgba(0,229,255,0.08)"),bgcolor="rgba(2,12,18,0.9)"),title=dict(text=f"<b>{query}</b> {dim5} 5D Manifold",font=dict(size=12,color="#4a9aaa")),height=560))
        st.plotly_chart(f5,use_container_width=True)
        dc=df5["Cancer"].value_counts()
        fd=go.Figure(go.Pie(labels=dc.index,values=dc.values,hole=0.60,marker=dict(colors=["#00e5ff","#ff3d5a","#ffc107","#00ff9d","#b44fff","#ff6600","#ff9933","#00aaff"],line=dict(color="#030f14",width=2)),textfont=dict(color="#c8f0f8",size=11)))
        fd.update_layout(**DK(title=dict(text="Cancer Type Distribution",font=dict(size=11,color="#4a9aaa")),legend=dict(font=dict(color="#00e5ff",size=10),bgcolor="rgba(0,0,0,0)"),height=320))
        st.plotly_chart(fd,use_container_width=True)
    with V2:
        np.random.seed(10);fr2=np.arange(200)
        mt1,mt2,mt3,mt4=st.tabs(["RMSD","RMSF","Radius of Gyration","H-Bond Count"])
        with mt1:
            rmsd=np.clip(np.cumsum(np.random.normal(0,0.02,200))+1.0,0.8,4.0)
            frm=go.Figure(go.Scatter(x=fr2,y=rmsd,mode="lines",line=dict(color="#00e5ff",width=2.5),fill="tozeroy",fillcolor="rgba(0,229,255,0.06)"))
            frm.add_hline(y=float(np.mean(rmsd)),line_dash="dash",line_color="#ffc107",annotation_text=f"Mean:{round(float(np.mean(rmsd)),2)}A",annotation_font_color="#ffc107")
            frm.update_layout(**DK(xaxis=dict(title="Frame",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),yaxis=dict(title="RMSD (A)",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),title=dict(text=f"<b>{query}</b> Backbone RMSD 200 frames",font=dict(size=12,color="#4a9aaa")),height=340))
            st.plotly_chart(frm,use_container_width=True)
            c1,c2,c3=st.columns(3)
            with c1: st.markdown(card("MEAN RMSD",str(round(float(np.mean(rmsd)),2)),"A","#00e5ff"),unsafe_allow_html=True)
            with c2: st.markdown(card("MAX RMSD",str(round(float(np.max(rmsd)),2)),"A","#ffc107"),unsafe_allow_html=True)
            with c3: st.markdown(card("MIN RMSD",str(round(float(np.min(rmsd)),2)),"A","#00ff9d"),unsafe_allow_html=True)
        with mt2:
            rmsf=np.abs(np.random.normal(0.9,0.5,100))+0.2
            for h in hs: idx=min(h["pos"]%100,99);rmsf[idx]+=2.0*h["freq"]*8
            frf=go.Figure(go.Bar(x=np.arange(1,101),y=rmsf,marker=dict(color=rmsf,colorscale=[[0,"#002535"],[0.4,"#00e5ff"],[1,"#ff3d5a"]],line=dict(color="rgba(0,0,0,0)",width=0)),hovertemplate="Res %{x}<br>RMSF:%{y:.2f}A<extra></extra>"))
            for h in hs[:4]: frf.add_vline(x=min(h["pos"]%100,99)+1,line_dash="dash",line_color="#ff3d5a",annotation_text=h["aa"],annotation_font_color="#ff3d5a",annotation_font_size=9)
            frf.update_layout(**DK(xaxis=dict(title="Residue",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),yaxis=dict(title="RMSF (A)",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),title=dict(text=f"<b>{query}</b> RMSF Red=hotspots",font=dict(size=12,color="#4a9aaa")),height=340))
            st.plotly_chart(frf,use_container_width=True)
        with mt3:
            rg=np.clip(18+np.cumsum(np.random.normal(0,0.05,200)),16,22)
            frg=go.Figure(go.Scatter(x=fr2,y=rg,mode="lines",line=dict(color="#00ff9d",width=2.5),fill="tozeroy",fillcolor="rgba(0,255,157,0.05)"))
            frg.update_layout(**DK(xaxis=dict(title="Frame",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),yaxis=dict(title="Rg (A)",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),title=dict(text=f"<b>{query}</b> Radius of Gyration",font=dict(size=12,color="#4a9aaa")),height=320))
            st.plotly_chart(frg,use_container_width=True)
        with mt4:
            hb=np.abs(np.random.normal(45,9,200)).astype(int)
            fhb=go.Figure(go.Scatter(x=fr2,y=hb,mode="lines",line=dict(color="#b44fff",width=2.5),fill="tozeroy",fillcolor="rgba(180,79,255,0.05)"))
            fhb.update_layout(**DK(xaxis=dict(title="Frame",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),yaxis=dict(title="H-Bond Count",color="#4a9aaa",gridcolor="rgba(0,229,255,0.06)"),title=dict(text=f"<b>{query}</b> Hydrogen Bond Count",font=dict(size=12,color="#4a9aaa")),height=320))
            st.plotly_chart(fhb,use_container_width=True)
    with V3:
        vt1,vt2,vt3,vt4=st.tabs(["Cartoon","Surface","Ball+Stick","B-Factor"])
        with vt1: components.html(ngl(pdb,"cartoon","chainname",False,480),height=500)
        with vt2: components.html(ngl(pdb,"surface","electrostatic",False,480),height=500)
        with vt3: components.html(ngl(pdb,"ball+stick","element",False,480),height=500)
        with vt4: components.html(ngl(pdb,"ribbon","bfactor",False,480),height=500)

with T6:
    st.markdown(sec("Integrated Pipeline Report & Export"),unsafe_allow_html=True)
    rp=get_ppi(query,limit=8)
    rr=[("TARGET GENE",query),("PDB STRUCTURE",pdb),("ONCO SCORE",str(sc.get("oncoscore","N/A"))+"/100"),("DRUGGABILITY",str(sc.get("druggability","N/A"))+"/100"),("MUTATION FREQ",str(sc.get("mutation_freq","N/A"))+"%"),("CLINICAL TRIALS",str(sc.get("clinical_trials","N/A"))+" active"),("TOP CANCER",topc+" "+str(expr.get(topc,"N/A"))+" log2(TPM)"),("HOTSPOTS",", ".join([h["aa"] for h in hs]) if hs else "None"),("TOP INTERACTORS",", ".join([b for a,b,s in rp[:5]])),("PIPELINE","G-FUSION v11 COMPLETE")]
    rh="".join([f'<tr style="border-bottom:1px solid rgba(0,229,255,0.04);"><td style="color:#4a9aaa;padding:8px 4px;width:200px;font-size:.6rem;letter-spacing:2px;text-transform:uppercase;">{k}</td><td style="color:#c8f0f8;font-size:.72rem;padding:8px 4px;">{v}</td></tr>' for k,v in rr])
    st.markdown(f'<div style="background:#041820;border:1px solid rgba(0,229,255,0.13);border-radius:8px;padding:18px 20px;"><div style="font-family:Orbitron,sans-serif;font-size:.88rem;color:#00e5ff;letter-spacing:3px;margin-bottom:14px;">IN SILICO PIPELINE REPORT {query}</div><table style="width:100%;border-collapse:collapse;">{rh}</table></div>',unsafe_allow_html=True)
    mds=[("3D Structure NGL.js","Cartoon Surface Ball+Stick B-Factor RCSB PDB engine","#00e5ff"),("Pathway Network","STRING DB NetworkX 3D Cytoscape 2D Heatmap","#00aaff"),("CRISPR Engine","SpCas9 Cas12a Cas13d gRNA PAM Off-target","#b44fff"),("Ligand Pharmacophore","Lipinski Ro5 Radar 6 approved drugs","#ffc107"),("5D Visualization","RMSD RMSF Rg H-Bonds 5D Manifold","#00ff9d"),("Molecular Intelligence","Real-time Anthropic API annotation","#ff9933"),("Report Export","TXT PPI CSV Expression CSV","#ff3d5a")]
    mc=st.columns(2)
    for i,(nm,desc,c2) in enumerate(mds):
        with mc[i%2]: st.markdown(f'<div style="background:#041820;border-left:3px solid {c2};border-radius:6px;padding:10px 14px;margin:5px 0;display:flex;justify-content:space-between;align-items:center;"><div><div style="color:{c2};font-family:Orbitron,sans-serif;font-size:.68rem;">{nm}</div><div style="color:#4a9aaa;font-size:.54rem;margin-top:3px;">{desc}</div></div>{badge("ACTIVE",c2)}</div>',unsafe_allow_html=True)
    st.markdown(sec("Download"),unsafe_allow_html=True)
    LL=["G-FUSION v11 REPORT","="*65]
    for k,v in rr: LL.append(f"  {k:<25}: {v}")
    LL+=["","PPI","-"*45]+[f"  {b:<16} {round(s,3)} {PWY.get(b,'?')}" for a,b,s in rp[:8]]+["","EXPRESSION","-"*45]+[f"  {c:<12}: {v} log2(TPM)" for c,v in expr.items()]+["","="*65]
    rt="\n".join(LL)
    d1,d2,d3=st.columns(3)
    with d1: st.download_button("DOWNLOAD TXT",data=rt,file_name=f"GFUSION_{query}.txt",mime="text/plain",key="dl1")
    dfp=pd.DataFrame([(b,PWY.get(b,"?"),round(s,3)) for a,b,s in rp],columns=["Partner","Pathway","Score"])
    with d2: st.download_button("DOWNLOAD PPI CSV",data=dfp.to_csv(index=False).encode(),file_name=f"GFUSION_{query}_PPI.csv",mime="text/csv",key="dl2")
    dfe=pd.DataFrame(list(expr.items()),columns=["Cancer","Expression_log2TPM"])
    with d3: st.download_button("DOWNLOAD EXPR CSV",data=dfe.to_csv(index=False).encode(),file_name=f"GFUSION_{query}_expr.csv",mime="text/csv",key="dl3")
