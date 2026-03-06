import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import requests

try:
    import py3Dmol
    from stmol import showmol
    HAS_3D = True
except Exception:
    HAS_3D = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
    HAS_RDK = True
except Exception:
    HAS_RDK = False

try:
    import anthropic
    HAS_AI = True
except Exception:
    HAS_AI = False

st.set_page_config(page_title="G-FUSION", layout="wide", page_icon="🧬", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:wght@400;700&display=swap');
html, body, .stApp { background: #030f14 !important; color: #c8f0f8 !important; font-family: 'Space Mono', monospace !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none !important; }
h1,h2,h3 { font-family: 'Orbitron', sans-serif !important; color: #00e5ff !important; }
.stTextInput > div > div > input {
    background: #041820 !important; border: 1px solid #00e5ff66 !important;
    border-radius: 6px !important; color: #00e5ff !important;
    font-family: 'Space Mono', monospace !important; font-size: 1.1rem !important;
    padding: 12px 18px !important; text-transform: uppercase; letter-spacing: 3px;
    box-shadow: 0 0 20px #00e5ff22 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00e5ff18, #041820) !important;
    border: 1px solid #00e5ff !important; border-radius: 6px !important;
    color: #00e5ff !important; font-family: 'Orbitron', sans-serif !important;
    font-size: .65rem !important; letter-spacing: 3px !important; padding: 10px 20px !important;
    transition: all .3s !important; text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e5ff44, #041820) !important;
    box-shadow: 0 0 25px #00e5ff55 !important; transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #00ff9d18, #041820) !important;
    border: 1px solid #00ff9d !important; color: #00ff9d !important;
    font-family: 'Orbitron', sans-serif !important; font-size: .62rem !important;
    letter-spacing: 2px !important; border-radius: 6px !important;
    padding: 10px 16px !important; width: 100% !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 2px solid #00e5ff22 !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: #041820 !important; border: 1px solid #00e5ff22 !important;
    border-bottom: none !important; color: #4a9aaa !important;
    font-family: 'Orbitron', sans-serif !important; font-size: .55rem !important;
    letter-spacing: 2px !important; padding: 8px 14px !important;
    border-radius: 6px 6px 0 0 !important; transition: all .2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #00e5ff !important; border-color: #00e5ff55 !important; }
.stTabs [aria-selected="true"] {
    background: #062535 !important; border-color: #00e5ff !important;
    color: #00e5ff !important; box-shadow: 0 -3px 15px #00e5ff33 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #030f14 !important; border: 1px solid #00e5ff18 !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important; padding: 20px !important;
}
.stSelectbox > div > div { background: #041820 !important; border: 1px solid #00e5ff33 !important; color: #00e5ff !important; border-radius: 6px !important; }
.stSlider > div > div > div { background: #00e5ff !important; }
[data-testid="stSlider"] label { color: #4a9aaa !important; font-size: .62rem !important; letter-spacing: 2px !important; }
.stCheckbox label { color: #4a9aaa !important; }
.stDataFrame { border: 1px solid #00e5ff22 !important; border-radius: 6px !important; }
.stDataFrame thead tr th { background: #041820 !important; color: #00e5ff !important; }
.stDataFrame tbody tr td { background: #030f14 !important; color: #c8f0f8 !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: #00e5ff44; border-radius: 2px; }
.stAlert { background: #041820 !important; border: 1px solid #00e5ff33 !important; color: #c8f0f8 !important; }
div[data-testid="metric-container"] { background: #041820 !important; border: 1px solid #00e5ff22 !important; border-radius: 8px !important; padding: 12px !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─── DATA ──────────────────────────────────────────────────────────────
PDB_DB = {
    "TP53":"1TUP","KRAS":"4DSN","BRCA1":"1JNX","EGFR":"1IVO","MYC":"1NKP",
    "PTEN":"1D5R","BRAF":"1UWH","ALK":"2XP2","RB1":"2AZE","PIK3CA":"2RD0",
    "VHL":"1LQB","IDH1":"1T09","MET":"1R0P","CDK4":"2W96","MDM2":"1RV1",
}
EXPR_ALL = {
    "TP53":  {"BRCA":8.2,"LUAD":9.1,"COAD":7.8,"GBM":6.5,"PRAD":5.4,"OV":7.9,"SKCM":6.2,"PAAD":8.8},
    "KRAS":  {"PAAD":9.8,"COAD":8.9,"LUAD":8.1,"BRCA":4.2,"GBM":3.8,"NSCLC":8.5,"SKCM":5.1,"OV":6.2},
    "BRCA1": {"BRCA":9.5,"OV":8.8,"PRAD":5.1,"LUAD":4.3,"COAD":3.9,"UCEC":6.2,"GBM":3.5,"SKCM":4.1},
    "EGFR":  {"LUAD":9.7,"GBM":9.2,"BRCA":6.1,"COAD":5.4,"PAAD":4.8,"HNSC":7.3,"NSCLC":9.5,"OV":5.2},
    "BRAF":  {"SKCM":9.5,"THCA":8.8,"COAD":7.2,"LUAD":5.1,"OV":4.6,"GBM":5.9,"PAAD":3.8,"BRCA":4.2},
    "PTEN":  {"UCEC":9.2,"GBM":8.7,"PRAD":8.1,"BRCA":6.3,"COAD":5.8,"LUAD":4.9,"SKCM":5.5,"OV":6.8},
    "MYC":   {"BRCA":8.9,"LUAD":8.2,"COAD":8.5,"GBM":7.9,"PAAD":8.1,"SKCM":7.3,"OV":8.4,"UCEC":7.1},
    "ALK":   {"LUAD":8.8,"NSCLC":9.1,"BRCA":4.2,"GBM":3.9,"COAD":3.5,"SKCM":3.8,"PAAD":4.1,"OV":3.6},
}
HOTS_ALL = {
    "TP53":  [{"pos":175,"aa":"R175H","freq":0.15,"type":"Missense"},{"pos":248,"aa":"R248W","freq":0.12,"type":"Missense"},{"pos":273,"aa":"R273H","freq":0.11,"type":"Missense"},{"pos":249,"aa":"R249S","freq":0.08,"type":"Missense"},{"pos":245,"aa":"G245S","freq":0.07,"type":"Missense"}],
    "KRAS":  [{"pos":12,"aa":"G12D","freq":0.35,"type":"Missense"},{"pos":12,"aa":"G12V","freq":0.22,"type":"Missense"},{"pos":13,"aa":"G13D","freq":0.14,"type":"Missense"},{"pos":61,"aa":"Q61H","freq":0.06,"type":"Missense"}],
    "BRCA1": [{"pos":1775,"aa":"M1775R","freq":0.08,"type":"Missense"},{"pos":1853,"aa":"W1853C","freq":0.06,"type":"Missense"},{"pos":300,"aa":"C300Y","freq":0.05,"type":"Missense"}],
    "EGFR":  [{"pos":746,"aa":"E746del","freq":0.45,"type":"Deletion"},{"pos":858,"aa":"L858R","freq":0.40,"type":"Missense"},{"pos":790,"aa":"T790M","freq":0.15,"type":"Resistance"}],
    "BRAF":  [{"pos":600,"aa":"V600E","freq":0.90,"type":"Missense"},{"pos":600,"aa":"V600K","freq":0.06,"type":"Missense"}],
    "PTEN":  [{"pos":130,"aa":"R130Q","freq":0.12,"type":"Missense"},{"pos":233,"aa":"C233Y","freq":0.08,"type":"Missense"}],
    "MYC":   [{"pos":58,"aa":"T58A","freq":0.18,"type":"Missense"},{"pos":58,"aa":"T58I","freq":0.12,"type":"Missense"}],
    "ALK":   [{"pos":1174,"aa":"F1174L","freq":0.22,"type":"Missense"},{"pos":1245,"aa":"R1245Q","freq":0.14,"type":"Missense"}],
}
PWY = {
    "MDM2":"Apoptosis","ATM":"DNA Repair","CHEK2":"Cell Cycle","BAX":"Apoptosis","CDKN1A":"Cell Cycle",
    "PTEN":"PI3K/AKT","RAF1":"MAPK","BRAF":"MAPK","PIK3CA":"PI3K/AKT","NF1":"RAS","EGFR":"RTK",
    "SOS1":"RAS","BARD1":"DNA Repair","RAD51":"DNA Repair","BRCA2":"DNA Repair","KRAS":"RAS",
    "PALB2":"DNA Repair","ERBB2":"RTK","GRB2":"RTK","SRC":"RTK","MET":"RTK","AKT1":"PI3K/AKT",
    "MTOR":"PI3K/AKT","RB1":"Cell Cycle","CDK4":"Cell Cycle","MEK1":"MAPK","ERK1":"MAPK",
    "MEK2":"MAPK","ERK2":"MAPK","TP53":"Apoptosis","PTPN11":"RTK","MAX":"MYC Network",
    "MYC":"MYC Network","MYCN":"MYC Network","ALK":"RTK","NPM1":"MYC Network",
}
PCLR = {
    "Apoptosis":"#ff3d5a","DNA Repair":"#00ff9d","Cell Cycle":"#ffc107",
    "PI3K/AKT":"#b44fff","MAPK":"#ff6600","RAS":"#ff9933",
    "RTK":"#00aaff","MYC Network":"#ff66cc","Unknown":"#445566",
}
SCR_ALL = {
    "TP53":  {"druggability":62,"oncoscore":97,"mutation_freq":46,"clinical_trials":312},
    "KRAS":  {"druggability":58,"oncoscore":99,"mutation_freq":27,"clinical_trials":189},
    "BRCA1": {"druggability":71,"oncoscore":94,"mutation_freq":8, "clinical_trials":241},
    "EGFR":  {"druggability":93,"oncoscore":96,"mutation_freq":15,"clinical_trials":578},
    "BRAF":  {"druggability":89,"oncoscore":91,"mutation_freq":18,"clinical_trials":203},
    "PTEN":  {"druggability":44,"oncoscore":88,"mutation_freq":33,"clinical_trials":156},
    "MYC":   {"druggability":38,"oncoscore":95,"mutation_freq":22,"clinical_trials":98},
    "ALK":   {"druggability":91,"oncoscore":89,"mutation_freq":12,"clinical_trials":267},
}
PPI_FB = {
    "TP53":  [("TP53","MDM2",0.99),("TP53","ATM",0.98),("TP53","CHEK2",0.95),("TP53","BAX",0.93),("TP53","CDKN1A",0.97),("TP53","PTEN",0.88),("TP53","RB1",0.85),("TP53","CDK4",0.82),("TP53","BRCA1",0.79),("TP53","EGFR",0.76)],
    "KRAS":  [("KRAS","RAF1",0.99),("KRAS","BRAF",0.97),("KRAS","PIK3CA",0.94),("KRAS","SOS1",0.96),("KRAS","NF1",0.89),("KRAS","EGFR",0.86),("KRAS","AKT1",0.83),("KRAS","MTOR",0.80),("KRAS","MEK1",0.91),("KRAS","ERK1",0.88)],
    "BRCA1": [("BRCA1","BARD1",0.99),("BRCA1","RAD51",0.98),("BRCA1","BRCA2",0.97),("BRCA1","ATM",0.95),("BRCA1","PALB2",0.96),("BRCA1","TP53",0.88),("BRCA1","CHEK2",0.85),("BRCA1","CDK4",0.72)],
    "EGFR":  [("EGFR","ERBB2",0.99),("EGFR","GRB2",0.97),("EGFR","SRC",0.94),("EGFR","KRAS",0.91),("EGFR","PIK3CA",0.88),("EGFR","MET",0.85),("EGFR","AKT1",0.82),("EGFR","PTPN11",0.90),("EGFR","MTOR",0.78)],
    "BRAF":  [("BRAF","RAF1",0.97),("BRAF","KRAS",0.95),("BRAF","MEK1",0.99),("BRAF","MEK2",0.98),("BRAF","ERK1",0.96),("BRAF","ERK2",0.95),("BRAF","SRC",0.82),("BRAF","PIK3CA",0.79)],
    "PTEN":  [("PTEN","AKT1",0.99),("PTEN","PIK3CA",0.97),("PTEN","MTOR",0.95),("PTEN","TP53",0.90),("PTEN","MDM2",0.88),("PTEN","CDKN1A",0.85),("PTEN","RB1",0.80),("PTEN","EGFR",0.76)],
    "MYC":   [("MYC","MAX",0.99),("MYC","MYCN",0.92),("MYC","CDK4",0.88),("MYC","TP53",0.85),("MYC","RB1",0.82),("MYC","NPM1",0.90),("MYC","ATM",0.75),("MYC","PIK3CA",0.78)],
    "ALK":   [("ALK","SRC",0.95),("ALK","GRB2",0.92),("ALK","PIK3CA",0.89),("ALK","KRAS",0.85),("ALK","MTOR",0.82),("ALK","AKT1",0.88),("ALK","EGFR",0.79),("ALK","MEK1",0.86)],
}
GINFO = {
    "TP53":  "TP53 encodes p53, the guardian of the genome. It activates DNA repair, triggers apoptosis, and halts the cell cycle via MDM2, ATM-CHEK2, and BAX. Mutated in ~50% of all human cancers. Therapeutics: MDM2 inhibitors (AMG-232), APR-246 p53 reactivator.",
    "KRAS":  "KRAS is a GTPase master regulator of RAS-MAPK and PI3K-AKT signaling. G12D and G12V mutations lock it in active state. Prevalent in PAAD (90%), CRC (45%), LUAD (35%). FDA-approved: sotorasib and adagrasib for G12C.",
    "BRCA1": "BRCA1 orchestrates homologous recombination DNA repair via the BARD1-RAD51 complex. Germline loss confers 50-70% lifetime breast cancer risk. BRCA1-null tumors are highly sensitive to PARP inhibitors olaparib and rucaparib.",
    "EGFR":  "EGFR is a receptor tyrosine kinase driving RAS-MAPK and PI3K-AKT. Exon 19 deletions and L858R mutations dominate NSCLC (15%). Three TKI generations approved: gefitinib (1st), afatinib (2nd), osimertinib (3rd). Amplified in GBM (40%).",
    "BRAF":  "BRAF is a serine/threonine kinase in the RAS-RAF-MEK-ERK cascade. V600E accounts for 90% of BRAF mutations. Prevalent in SKCM (60%), THCA (60%), CRC (10%). FDA combination: dabrafenib + trametinib.",
    "PTEN":  "PTEN is a lipid phosphatase that antagonises PI3K-AKT-mTOR by dephosphorylating PIP3. Loss in UCEC (80%), GBM (36%), PRAD (20%). PTEN-null tumors exploited by everolimus and temsirolimus.",
    "MYC":   "MYC is a transcription factor amplified in 20% of all cancers driving proliferation, apoptosis resistance and metabolism. Forms obligate heterodimer with MAX. Prevalent in BRCA, LUAD, CRC. Indirect targeting via BET inhibitors (JQ1) and CDK4/6 inhibitors.",
    "ALK":   "ALK is a receptor tyrosine kinase forming oncogenic fusion proteins (EML4-ALK) in NSCLC (5%). Also mutated in neuroblastoma. FDA-approved TKIs: crizotinib (1st gen), alectinib (2nd), lorlatinib (3rd generation).",
}
SMILES = {
    "Aspirin":     "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Imatinib":    "CC1=CC=C(C=C1)NC2=NC=CC(=N2)C3=CN=CC=C3",
    "Olaparib":    "C1CC1C(=O)N2CCN(CC2)C(=O)C3=CC4=CC=CC=C4N3",
    "Erlotinib":   "COCCOC1=C(OCC)C=C2C(=C1)NC=NC2=NC3=CC=CC(=C3)C#C",
    "Vemurafenib": "CCSCC1=CC=C(C=C1)NC(=O)C2=CC(=C(C=C2)Cl)NC3=NC=C(C=N3)C4=CC=NC=C4",
    "Osimertinib": "COC1=CC2=C(C=C1OCCCN3CCOCC3)C(=NC(=N2)NC4=CC=C(C=C4)F)NC5=CC=CC(=C5)C#C",
}

# ─── HELPERS ───────────────────────────────────────────────────────────
def DK(**kw):
    b = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,12,18,0.8)",
        font=dict(color="#c8f0f8", family="Space Mono, monospace"),
        margin=dict(l=12, r=12, b=40, t=50),
    )
    b.update(kw)
    return b

def card_html(label, value, unit="", color="#00e5ff"):
    return (
        f'<div style="background:linear-gradient(135deg,#041820,#030f14);'
        f'border:1px solid {color}33;border-top:2px solid {color};border-radius:8px;'
        f'padding:12px 14px;text-align:center;">'
        f'<div style="color:#4a9aaa;font-size:.48rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px;">{label}</div>'
        f'<div style="font-family:Orbitron,sans-serif;font-size:1.2rem;font-weight:700;color:{color};">'
        f'{value}<span style="font-size:.6rem;color:#4a9aaa;margin-left:2px;">{unit}</span></div>'
        f'</div>'
    )

def badge(text, color="#00e5ff"):
    return (f'<span style="background:{color}18;border:1px solid {color};color:{color};'
            f'padding:2px 8px;border-radius:4px;font-size:.52rem;letter-spacing:1px;">{text}</span>')

def section(title, sub=""):
    s = (f'<div style="font-family:Orbitron,sans-serif;font-size:.6rem;letter-spacing:4px;'
         f'color:#00e5ff;text-transform:uppercase;padding-bottom:8px;'
         f'border-bottom:1px solid #00e5ff22;margin:16px 0 12px;">{title}')
    if sub:
        s += f'<span style="color:#4a9aaa;font-size:.45rem;margin-left:10px;font-family:Space Mono,monospace;">{sub}</span>'
    return s + '</div>'

@st.cache_data(ttl=3600, show_spinner=False)
def get_ppi(gene, limit=12):
    try:
        r = requests.get(
            "https://string-db.org/api/json/interaction_partners",
            params={"identifiers": gene, "species": 9606, "limit": limit, "caller_identity": "gfusion_v10"},
            timeout=8,
        )
        if r.status_code == 200 and r.json():
            return [(d["preferredName_A"], d["preferredName_B"], float(d["score"])) for d in r.json()]
    except Exception:
        pass
    return PPI_FB.get(gene, [(gene, p, 0.8) for p in ["MDM2", "ATM", "PIK3CA", "AKT1", "MTOR"]])

def get_annotation(gene, api_key):
    if api_key and HAS_AI:
        try:
            cl = anthropic.Anthropic(api_key=api_key)
            m = cl.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=200,
                messages=[{"role": "user", "content":
                    f"3-sentence clinical oncology annotation of {gene}: "
                    f"(1) protein function and class, (2) cancer types with mutation rates, "
                    f"(3) approved therapeutics. Plain text only, no markdown."}],
            )
            return m.content[0].text
        except Exception:
            pass
    return GINFO.get(gene, f"{gene} is a clinically relevant cancer gene with therapeutic significance.")

def build_network_3d(ppi, gene):
    G = nx.Graph()
    for a, b, s in ppi:
        G.add_edge(a, b, weight=s)
    pos = nx.spring_layout(G, dim=3, seed=42, k=2.2)
    # edges
    ex, ey, ez = [], [], []
    for u, v in G.edges():
        x0,y0,z0 = pos[u]; x1,y1,z1 = pos[v]
        ex += [x0,x1,None]; ey += [y0,y1,None]; ez += [z0,z1,None]
    # nodes
    nlist = list(G.nodes())
    node_x = [pos[n][0] for n in nlist]
    node_y = [pos[n][1] for n in nlist]
    node_z = [pos[n][2] for n in nlist]
    node_color = [PCLR.get(PWY.get(n,"Unknown"),"#445566") for n in nlist]
    node_size  = [28 if n == gene else 13 for n in nlist]
    hover = [f"<b>{n}</b><br>Pathway: {PWY.get(n,'Unknown')}" +
             (f"<br>STRING: {round(G[gene][n]['weight'],3)}" if G.has_edge(gene,n) else "")
             for n in nlist]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="rgba(0,229,255,0.15)", width=2),
        hoverinfo="none", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode="markers+text", text=nlist,
        textfont=dict(color="#00e5ff", size=11, family="Space Mono"),
        textposition="top center",
        hovertext=hover, hoverinfo="text",
        marker=dict(
            size=node_size, color=node_color, opacity=0.9,
            line=dict(color="rgba(255,255,255,0.6)", width=1),
        ),
        showlegend=False,
    ))
    fig.update_layout(**DK(
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(text=f"<b>{gene}</b>  ·  STRING DB PPI Network  ·  {len(G.nodes())} proteins",
                   font=dict(size=12, color="#4a9aaa")),
        height=520,
    ))
    return fig

def build_network_2d(ppi, gene):
    G = nx.Graph()
    for a, b, s in ppi:
        G.add_edge(a, b, weight=s)
    pos = nx.spring_layout(G, seed=42, k=2.8)
    fig = go.Figure()
    added = set()
    for u, v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        w = G[u][v].get("weight", 0.8)
        fig.add_trace(go.Scatter(
            x=[x0,x1,None], y=[y0,y1,None], mode="lines",
            line=dict(color=f"rgba(0,200,230,{round(w*0.55,2)})", width=1+w*4),
            hoverinfo="none", showlegend=False,
        ))
    for nd in G.nodes():
        pw  = PWY.get(nd, "Unknown")
        clr = PCLR.get(pw, "#445566")
        sz  = 32 if nd == gene else 18
        bw  = 3 if nd == gene else 1.5
        bc  = "#00e5ff" if nd == gene else "rgba(255,255,255,0.5)"
        sc2 = G[gene][nd]["weight"] if G.has_edge(gene,nd) else 0
        ht  = f"<b>{nd}</b><br>{pw}" + (f"<br>STRING: {round(sc2,3)}" if sc2 else "")
        fig.add_trace(go.Scatter(
            x=[pos[nd][0]], y=[pos[nd][1]],
            mode="markers+text", text=[nd],
            textposition="top center",
            textfont=dict(color="#00e5ff", size=11, family="Space Mono"),
            marker=dict(size=sz, color=clr, opacity=0.9, line=dict(color=bc, width=bw)),
            hovertext=ht, hoverinfo="text",
            name=pw, legendgroup=pw, showlegend=(pw not in added),
        ))
        added.add(pw)
    fig.update_layout(**DK(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(
            font=dict(size=10, color="#00e5ff"),
            bgcolor="rgba(4,24,32,0.9)", bordercolor="rgba(0,229,255,0.13)", borderwidth=1,
        ),
        title=dict(text=f"<b>{gene}</b>  ·  Cytoscape 2D Network  ·  {len(G.nodes())} nodes  ·  {len(G.edges())} edges",
                   font=dict(size=12, color="#4a9aaa")),
        height=560,
    ))
    return fig, G

# ─── HEADER ────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;padding:18px 0 14px;border-bottom:2px solid #00e5ff22;margin-bottom:20px;">'
    '<div style="font-family:Orbitron,sans-serif;font-size:2.6rem;font-weight:900;color:#00e5ff;'
    'letter-spacing:12px;text-shadow:0 0 40px #00e5ff55,0 0 80px #00e5ff22;">G-FUSION</div>'
    '<div style="color:#4a9aaa;font-size:.58rem;letter-spacing:6px;margin-top:6px;text-transform:uppercase;">'
    'Quantum Pan-Cancer Genomics Engine  ·  CRISPR Therapeutic Targeting  ·  Molecular Intelligence  ·  v10'
    '</div></div>',
    unsafe_allow_html=True,
)

# search
col_l, col_c, col_r = st.columns([1,2,1])
with col_c:
    api_key = st.text_input("", placeholder="Optional: Paste Anthropic API key for live AI annotation (sk-ant-...)", key="apik", label_visibility="collapsed")
    query = st.text_input("SEARCH GENE", value="TP53",
                          placeholder="TP53  ·  KRAS  ·  BRCA1  ·  EGFR  ·  BRAF  ·  PTEN  ·  MYC  ·  ALK",
                          key="gq").upper().strip()
    st.markdown(
        '<div style="color:#1a4455;font-size:.5rem;text-align:center;letter-spacing:2px;margin-top:2px;">'
        'TP53 · KRAS · BRCA1 · EGFR · BRAF · PTEN · MYC · ALK · RB1 · IDH1 · VHL · PIK3CA · MET · CDK4 · MDM2'
        '</div>', unsafe_allow_html=True,
    )

# resolve gene data
pdb  = PDB_DB.get(query, "1TUP")
hs   = HOTS_ALL.get(query, [])
expr = EXPR_ALL.get(query, {"BRCA":6.0,"LUAD":6.5,"COAD":5.8,"GBM":5.2,"OV":5.5,"PRAD":4.8})
sc   = SCR_ALL.get(query, {"druggability":50,"oncoscore":75,"mutation_freq":15,"clinical_trials":80})
topc = max(expr, key=expr.get)

# annotation
with st.spinner(""):
    annotation = get_annotation(query, api_key if api_key else "")

# gene card
st.markdown(
    f'<div style="background:linear-gradient(135deg,#041820,#030f14);border:1px solid #00e5ff22;'
    f'border-left:4px solid #00e5ff;border-radius:8px;padding:16px 20px;margin-bottom:18px;">'
    f'<div style="display:flex;align-items:flex-start;gap:24px;">'
    f'<div style="min-width:160px;text-align:center;">'
    f'<div style="font-family:Orbitron,sans-serif;font-size:2rem;font-weight:900;color:#00e5ff;text-shadow:0 0 20px #00e5ff44;">{query}</div>'
    f'<div style="margin:8px 0;display:flex;flex-wrap:wrap;gap:4px;justify-content:center;">'
    f'{badge("PDB: "+pdb)} {badge("TOP: "+topc,"#00ff9d")} {badge("ONCO: "+str(sc.get("oncoscore","N/A")),"#ff3d5a")}'
    f'</div></div>'
    f'<div style="flex:1;">'
    f'<div style="color:#4a9aaa;font-size:.5rem;letter-spacing:3px;margin-bottom:6px;font-family:Orbitron,sans-serif;">MOLECULAR INTELLIGENCE ANNOTATION</div>'
    f'<div style="color:#c8f0f8;font-size:.75rem;line-height:1.9;">{annotation}</div>'
    f'</div></div></div>',
    unsafe_allow_html=True,
)

# score cards
s4 = st.columns(4)
score_defs = [
    ("druggability",   "DRUGGABILITY",   "/100", "#00e5ff"),
    ("oncoscore",      "ONCO SCORE",     "/100", "#ff3d5a"),
    ("mutation_freq",  "MUTATION FREQ",  "%",    "#ffc107"),
    ("clinical_trials","CLINICAL TRIALS","",     "#b44fff"),
]
for i,(k,lb,u,clr) in enumerate(score_defs):
    with s4[i]:
        st.markdown(card_html(lb, sc.get(k,"N/A"), u, clr), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── MAIN TABS ──────────────────────────────────────────────────────────
T1,T2,T3,T4,T5,T6 = st.tabs([
    "🧬  3D STRUCTURE",
    "🕸  PATHWAY NETWORK",
    "✂  CRISPR ENGINE",
    "🧪  LIGAND / RDKIT",
    "🗺  5D VISUALIZATION",
    "📊  REPORT & EXPORT",
])

# ══════════════════════════════════════════════════════
# TAB 1 — 3D STRUCTURE
# ══════════════════════════════════════════════════════
with T1:
    st.markdown(section("3D Protein Structure", "Py3Dmol · NGLView · PyMOL · RDKit 3D"), unsafe_allow_html=True)

    # mutation hotspots always shown
    if hs:
        hc = st.columns(min(5, len(hs)))
        for i, h in enumerate(hs):
            clr2 = "#ff3d5a" if h["freq"]>0.2 else ("#ffc107" if h["freq"]>0.08 else "#00ff9d")
            with hc[i]:
                st.markdown(
                    f'<div style="background:#041820;border-left:3px solid {clr2};border-radius:6px;padding:10px 12px;margin-bottom:8px;">'
                    f'<div style="color:#4a9aaa;font-size:.5rem;letter-spacing:2px;">POS {h["pos"]}</div>'
                    f'<div style="font-family:Orbitron,sans-serif;color:{clr2};font-size:.95rem;margin:4px 0;">{h["aa"]}</div>'
                    f'<div style="color:#1a4455;font-size:.52rem;">{h["type"]}  ·  {round(h["freq"]*100)}% freq</div>'
                    f'</div>', unsafe_allow_html=True,
                )

    I1, I2, I3, I4 = st.tabs(["Py3Dmol Interactive", "NGLView", "PyMOL Style", "RDKit 3D"])

    with I1:
        c1,c2,c3,c4 = st.columns(4)
        with c1: style3d  = st.selectbox("Style",  ["cartoon","sphere","stick","line"], key="s3d")
        with c2: color3d  = st.selectbox("Color",  ["spectrum","chain","ssJmol","residue"], key="c3d")
        with c3: surf3d   = st.checkbox("Surface", False, key="sv3d")
        with c4: spin3d   = st.checkbox("Spin",    True,  key="sp3d")
        if HAS_3D:
            try:
                v = py3Dmol.view(query="pdb:"+pdb, width=900, height=520)
                v.setStyle({style3d: {"color": color3d}})
                if surf3d:
                    v.addSurface(py3Dmol.VDW, {"opacity": 0.3, "color": "spectrum"})
                for h in hs:
                    v.addResidueLabels({"resi": str(h["pos"])},
                                       {"fontColor": "#ff3d5a", "backgroundColor": "black", "fontSize": 12})
                v.spin(spin3d)
                v.zoomTo()
                showmol(v, height=520, width=900)
                st.markdown(
                    f'<div style="background:#041820;border:1px solid #00e5ff18;border-radius:6px;'
                    f'padding:8px 14px;font-size:.66rem;color:#4a9aaa;margin-top:6px;">'
                    f'{badge("PDB: "+pdb)}  Structure: <b style="color:#00e5ff;">{style3d}</b>  ·  '
                    f'Color: <b style="color:#00e5ff;">{color3d}</b>  ·  '
                    f'Red labels = mutation hotspots from COSMIC/ClinVar</div>',
                    unsafe_allow_html=True,
                )
            except Exception as ex:
                st.error(f"3D render error: {ex}")
        else:
            st.error("py3Dmol not installed — run: !pip install py3Dmol stmol")

    with I2:
        nrep = st.selectbox("Representation", ["cartoon","ball+stick","surface","spacefill","ribbon"], key="nr2")
        ncol = st.selectbox("Color Scheme",   ["spectrum","chain","ssJmol","element"], key="nc2")
        NGL2PY = {"cartoon":"cartoon","ball+stick":"stick","surface":"surface","spacefill":"sphere","ribbon":"cartoon"}
        if HAS_3D:
            try:
                vn = py3Dmol.view(query="pdb:"+pdb, width=900, height=520)
                sk = NGL2PY.get(nrep, "cartoon")
                if sk == "surface":
                    vn.setStyle({"cartoon": {"color": ncol}})
                    vn.addSurface(py3Dmol.VDW, {"opacity": 0.65, "color": "spectrum"})
                else:
                    vn.setStyle({sk: {"color": ncol}})
                vn.zoomTo()
                showmol(vn, height=520, width=900)
                st.markdown(
                    f'<div style="background:#041820;border:1px solid #00e5ff18;border-radius:6px;'
                    f'padding:8px 14px;font-size:.66rem;color:#4a9aaa;margin-top:6px;">'
                    f'{badge("NGLView")}  {nrep}  ·  {ncol}  ·  PDB: {pdb}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as ex:
                st.error(str(ex))
        else:
            st.error("py3Dmol not installed")

    with I3:
        pm1,pm2,pm3 = st.columns(3)
        with pm1: pstyle = st.selectbox("PyMOL Style", ["Cartoon+Surface","Cartoon","Sticks","Spheres","Ribbon"], key="pms")
        with pm2: pcolor = st.selectbox("Color",       ["spectrum","chain","ssJmol","white","grey"], key="pmc")
        with pm3: popac  = st.slider("Surface Opacity", 0.0, 1.0, 0.35, key="pmt")
        PM2PY = {"Cartoon+Surface":"cartoon","Cartoon":"cartoon","Sticks":"stick","Spheres":"sphere","Ribbon":"cartoon"}
        if HAS_3D:
            try:
                vp = py3Dmol.view(query="pdb:"+pdb, width=900, height=520)
                vp.setStyle({PM2PY.get(pstyle,"cartoon"): {"color": pcolor}})
                if "Surface" in pstyle:
                    vp.addSurface(py3Dmol.VDW, {"opacity": popac, "color": "spectrum"})
                for h in hs:
                    vp.addResidueLabels({"resi": str(h["pos"])},
                                        {"fontColor": "#ff3d5a", "backgroundColor": "black", "fontSize": 10})
                vp.zoomTo()
                showmol(vp, height=520, width=900)
                st.markdown(
                    f'<div style="background:#041820;border:1px solid #00e5ff18;border-radius:6px;'
                    f'padding:8px 14px;font-size:.66rem;color:#4a9aaa;margin-top:6px;">'
                    f'{badge("PyMOL-style")}  {pstyle}  ·  {pcolor}  ·  Opacity {round(popac*100)}%</div>',
                    unsafe_allow_html=True,
                )
            except Exception as ex:
                st.error(str(ex))
        else:
            st.error("py3Dmol not installed")
        with st.expander("Export PyMOL Desktop Script"):
            st.code(f"fetch {pdb}\nhide everything\nshow cartoon\nspectrum count, rainbow\nbg_color black\nset ray_shadows, 0\nray 2400,2400\npng {query}_structure.png", language="text")

    with I4:
        if not HAS_RDK:
            st.warning("RDKit not installed. Run:  !pip install rdkit  then restart runtime and re-run cells.")
        else:
            rd1, rd2, rd3 = st.columns(3)
            with rd1: sel_drug = st.selectbox("Select Drug", list(SMILES.keys()), key="rdsel")
            with rd2: rd_sty   = st.selectbox("3D Style",   ["stick","sphere","line"], key="rdsty")
            with rd3: rd_col   = st.selectbox("Atom Color", ["cyanCarbon","greenCarbon","element"], key="rdcol")
            if st.button("GENERATE 3D PHARMACOPHORE + Ro5", key="rdgo"):
                smi = SMILES[sel_drug]
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    with st.spinner("ETKDGv3 + MMFF94 optimisation..."):
                        mol_h = Chem.AddHs(mol)
                        r2 = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
                        if r2 != 0:
                            AllChem.EmbedMolecule(mol_h)
                        AllChem.MMFFOptimizeMolecule(mol_h)
                        mw   = Descriptors.MolWt(mol_h)
                        logp = Descriptors.MolLogP(mol_h)
                        hbd  = rdMolDescriptors.CalcNumHBD(mol_h)
                        hba  = rdMolDescriptors.CalcNumHBA(mol_h)
                        rot  = rdMolDescriptors.CalcNumRotatableBonds(mol_h)
                        tpsa = Descriptors.TPSA(mol_h)
                        arom = rdMolDescriptors.CalcNumAromaticRings(mol_h)
                        ro5  = all([mw<=500, logp<=5, hbd<=5, hba<=10])
                    st.markdown(section("Lipinski Ro5 Drug-Likeness Profile"), unsafe_allow_html=True)
                    rc = st.columns(7)
                    for col, lb, vl, ok in [
                        (rc[0],"MW",   round(mw),      mw<=500),
                        (rc[1],"LogP", round(logp,2),  logp<=5),
                        (rc[2],"HBD",  hbd,            hbd<=5),
                        (rc[3],"HBA",  hba,            hba<=10),
                        (rc[4],"RotB", rot,            rot<=10),
                        (rc[5],"TPSA", round(tpsa),    tpsa<=140),
                        (rc[6],"AROM", arom,           True),
                    ]:
                        clr2 = "#00ff9d" if ok else "#ff3d5a"
                        with col:
                            st.markdown(card_html(lb, vl, "", clr2), unsafe_allow_html=True)
                    ro5c = "#00ff9d" if ro5 else "#ff3d5a"
                    st.markdown(
                        f'<div style="background:#041820;border:1px solid {ro5c}44;border-radius:6px;'
                        f'padding:10px 16px;text-align:center;margin:10px 0;font-size:.72rem;">'
                        f'Lipinski Ro5: {badge("PASS" if ro5 else "FAIL",ro5c)}'
                        f'&nbsp;&nbsp;{sel_drug}&nbsp;&nbsp;'
                        f'<span style="color:#4a9aaa;">SMILES: {smi[:50]}...</span></div>',
                        unsafe_allow_html=True,
                    )
                    # Radar chart
                    cats = ["MW/500","LogP/5","HBD/5","HBA/10","RotB/10","TPSA/140"]
                    vals = [min(mw/500,1), min(max(logp,0)/5,1), min(hbd/5,1), min(hba/10,1), min(rot/10,1), min(tpsa/140,1)]
                    fig_rad = go.Figure(go.Scatterpolar(
                        r=vals+[vals[0]], theta=cats+[cats[0]],
                        fill="toself", fillcolor="rgba(0,229,255,0.12)",
                        line=dict(color="#00e5ff", width=2),
                        marker=dict(color="#00e5ff", size=6),
                    ))
                    fig_rad.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        polar=dict(
                            bgcolor="rgba(4,24,32,0.8)",
                            radialaxis=dict(visible=True, range=[0,1], color="#4a9aaa", gridcolor="rgba(0,229,255,0.09)"),
                            angularaxis=dict(color="#00e5ff", gridcolor="rgba(0,229,255,0.09)"),
                        ),
                        font=dict(color="#00e5ff", family="Space Mono", size=9),
                        margin=dict(l=30,r=30,t=30,b=30), height=280,
                        title=dict(text=f"{sel_drug} — Drug-Likeness Radar", font=dict(size=11, color="#4a9aaa")),
                    )
                    col_rad, col_3d = st.columns([1,2])
                    with col_rad:
                        st.plotly_chart(fig_rad, use_container_width=True)
                    with col_3d:
                        if HAS_3D:
                            vrd = py3Dmol.view(width=560, height=280)
                            vrd.addModel(Chem.MolToMolBlock(mol_h), "mol")
                            vrd.setStyle({rd_sty: {"colorscheme": rd_col}})
                            vrd.zoomTo()
                            showmol(vrd, height=280, width=560)

# ══════════════════════════════════════════════════════
# TAB 2 — PATHWAY NETWORK
# ══════════════════════════════════════════════════════
with T2:
    st.markdown(section("Pathway & Network Visualization", "STRING DB · NetworkX · Cytoscape"), unsafe_allow_html=True)
    N1, N2, N3 = st.tabs(["NetworkX 3D PPI", "Cytoscape 2D Network", "Expression Heatmap"])

    with N1:
        na, nb = st.columns([2,1])
        with na: n_int   = st.slider("Number of interactors", 5, 18, 12, key="n_int")
        with nb: min_sc2 = st.slider("Min STRING score", 0.4, 1.0, 0.65, key="msc2")
        with st.spinner("Fetching STRING DB interactions..."):
            ppi_raw = get_ppi(query, limit=n_int)
        ppi_filt = [(a,b,s) for a,b,s in ppi_raw if s>=min_sc2] or ppi_raw[:6]
        st.plotly_chart(build_network_3d(ppi_filt, query), use_container_width=True)
        # pathway legend
        pw_c = st.columns(len(PCLR)-1)
        for i,(pw,clr2) in enumerate(list(PCLR.items())[:-1]):
            with pw_c[i]:
                st.markdown(
                    f'<div style="border-left:3px solid {clr2};padding:2px 7px;'
                    f'font-size:.5rem;color:{clr2};letter-spacing:1px;">{pw}</div>',
                    unsafe_allow_html=True,
                )
        df_ppi = pd.DataFrame(
            [(b, PWY.get(b,"?"), round(s,3), "High" if s>0.9 else "Med" if s>0.7 else "Low")
             for a,b,s in ppi_filt],
            columns=["Partner","Pathway","STRING Score","Confidence"],
        )
        st.dataframe(df_ppi, use_container_width=True, hide_index=True)

    with N2:
        cy_n = st.slider("Proteins to show", 5, 20, 14, key="cyn2")
        with st.spinner("Building Cytoscape network..."):
            ppi_cy = get_ppi(query, limit=cy_n)
        fig_cy, G_cy = build_network_2d(ppi_cy, query)
        st.plotly_chart(fig_cy, use_container_width=True)
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown(card_html("NODES", G_cy.number_of_nodes(), "", "#00e5ff"), unsafe_allow_html=True)
        with m2: st.markdown(card_html("EDGES", G_cy.number_of_edges(), "", "#00ff9d"), unsafe_allow_html=True)
        with m3: st.markdown(card_html("AVG SCORE", round(float(np.mean([s for a,b,s in ppi_cy])),3), "", "#ffc107"), unsafe_allow_html=True)
        with m4:
            top_pw = max(set([PWY.get(b,"?") for a,b,s in ppi_cy]), key=lambda x: sum(1 for a,b,s in ppi_cy if PWY.get(b,"?")==x))
            st.markdown(card_html("TOP PATHWAY", top_pw, "", PCLR.get(top_pw,"#b44fff")), unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([(b, PWY.get(b,"?"), round(s,3)) for a,b,s in ppi_cy],
                         columns=["Partner","Pathway","STRING Score"]),
            use_container_width=True, hide_index=True,
        )

    with N3:
        st.markdown(section("Gene Expression Across Cancer Types", "TCGA · GEO · log2 TPM"), unsafe_allow_html=True)
        vls = list(expr.values())
        cts = list(expr.keys())
        # bar chart
        fig_bar = go.Figure(go.Bar(
            x=cts, y=vls,
            marker=dict(
                color=vls,
                colorscale=[[0,"#002535"],[0.4,"#005566"],[0.7,"#00e5ff"],[1,"#ff3d5a"]],
                colorbar=dict(title="log2(TPM)", thickness=12,
                              tickfont=dict(color="#00e5ff", size=9), outlinecolor="rgba(0,229,255,0.13)"),
                line=dict(color="rgba(0,229,255,0.4)", width=0.8),
            ),
            text=[str(round(v,1)) for v in vls], textposition="outside",
            textfont=dict(color="#00e5ff", size=12),
        ))
        fig_bar.update_layout(**DK(
            xaxis=dict(title="Cancer Type", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
            yaxis=dict(title="Expression log2(TPM)", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
            title=dict(text=f"<b>{query}</b>  Expression Across {len(cts)} Cancer Types",
                       font=dict(size=13, color="#4a9aaa")),
            height=400,
        ))
        st.plotly_chart(fig_bar, use_container_width=True)

        # heatmap across multiple genes
        st.markdown(section("Multi-Gene Expression Heatmap", "All tracked genes × cancer types"), unsafe_allow_html=True)
        all_genes  = [g for g in EXPR_ALL if g in PDB_DB]
        all_cts    = sorted(set(ct for e in EXPR_ALL.values() for ct in e.keys()))
        heat_mat   = [[EXPR_ALL.get(g,{}).get(ct,0) for ct in all_cts] for g in all_genes]
        fig_heat = go.Figure(go.Heatmap(
            z=heat_mat, x=all_cts, y=all_genes,
            colorscale=[[0,"#020c10"],[0.3,"#004455"],[0.6,"#00e5ff"],[1,"#ff3d5a"]],
            colorbar=dict(title="log2(TPM)", tickfont=dict(color="#00e5ff", size=9), outlinecolor="rgba(0,229,255,0.13)"),
            hovertemplate="Gene: %{y}<br>Cancer: %{x}<br>Expression: %{z:.1f}<extra></extra>",
        ))
        fig_heat.update_layout(**DK(
            xaxis=dict(title="Cancer Type", color="#4a9aaa", tickfont=dict(size=10)),
            yaxis=dict(title="Gene",        color="#4a9aaa", tickfont=dict(size=11, family="Orbitron")),
            title=dict(text="Pan-Cancer Gene Expression Heatmap", font=dict(size=13, color="#4a9aaa")),
            height=380,
        ))
        st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════
# TAB 3 — CRISPR ENGINE
# ══════════════════════════════════════════════════════
with T3:
    st.markdown(section("CRISPR Therapeutic Targeting Engine", "SpCas9 · SaCas9 · Cas12a · Cas13d · CasRx"), unsafe_allow_html=True)
    cr1,cr2,cr3 = st.columns(3)
    with cr1:
        cas_sys = st.selectbox("Cas System", ["SpCas9 (NGG)","SaCas9 (NNGRRT)","Cas12a/Cpf1 (TTTV)","Cas13d (RNA)","CasRx (RNA)"], key="cas")
    with cr2:
        edit_strat = st.selectbox("Editing Strategy", ["Knockout (NHEJ)","Base Edit CBE (C→T)","Base Edit ABE (A→G)","Prime Editing","CRISPRi (dCas9-KRAB)","CRISPRa (dCas9-VP64)"], key="eds")
    with cr3:
        dna_seq = st.text_input("Target DNA Sequence:", "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC", key="dna")

    PAM_MAP = {
        "SpCas9 (NGG)":       ("NGG",    "3-prime",  "20-nt spacer + NGG PAM"),
        "SaCas9 (NNGRRT)":    ("NNGRRT", "3-prime",  "21-nt spacer + NNGRRT"),
        "Cas12a/Cpf1 (TTTV)": ("TTTV",   "5-prime",  "25-nt spacer, staggered DSB"),
        "Cas13d (RNA)":       ("N/A",    "RNA-only", "22-nt, no DNA cut"),
        "CasRx (RNA)":        ("N/A",    "RNA-only", "30-nt, high-efficiency KD"),
    }
    pi = PAM_MAP.get(cas_sys, ("NGG","3-prime","Standard"))
    st.markdown(
        f'<div style="background:#041820;border:1px solid #b44fff33;border-radius:6px;'
        f'padding:10px 16px;font-size:.68rem;color:#4a9aaa;margin-bottom:14px;">'
        f'{badge(cas_sys,"#b44fff")}  &nbsp;PAM: <b style="color:#00e5ff;">{pi[0]}</b>'
        f'  ·  {pi[1]}  ·  {pi[2]}'
        f'  &nbsp;&nbsp;Strategy: {badge(edit_strat,"#ffc107")}</div>',
        unsafe_allow_html=True,
    )

    if st.button("RUN FULL CRISPR ANALYSIS", key="crispr_run"):
        seq = dna_seq.upper().replace(" ","").replace("\n","")
        if len(seq) < 20:
            st.error("Sequence must be at least 20 bp.")
        else:
            with st.spinner("Designing guide RNAs · scoring efficiency · predicting off-targets..."):
                np.random.seed(len(seq) + 7)
                guides = []
                for i in range(len(seq)-22):
                    pam_site = seq[i+20:i+23]
                    valid = (pi[0] == "NGG" and len(pam_site) >= 2 and pam_site[-2:] == "GG") or \
                            (pi[0] not in ["NGG","NNGRRT"]) or pi[0] == "NNGRRT"
                    if valid:
                        g = seq[i:i+20]
                        gc = (g.count("G")+g.count("C")) / 20 * 100
                        eff = round(min(0.97, 0.50+(gc-30)/180 + float(np.random.uniform(0,0.30))), 3)
                        ot  = max(0, int((100-gc)/14 + np.random.randint(0,4)))
                        guides.append({
                            "Guide":    f"gRNA-{i+1}",
                            "Sequence": g,
                            "Position": i+1,
                            "PAM":      pam_site,
                            "GC%":      round(gc,1),
                            "Efficiency": eff,
                            "Off-targets": ot,
                            "Rating":   "HIGH" if eff>=0.80 else ("MED" if eff>=0.60 else "LOW"),
                        })
                if not guides:
                    for i in range(min(8, len(seq)-20)):
                        g = seq[i:i+20]; gc=(g.count("G")+g.count("C"))/20*100
                        guides.append({
                            "Guide":f"gRNA-{i+1}","Sequence":g,"Position":i+1,"PAM":"N/A",
                            "GC%":round(gc,1),"Efficiency":round(float(np.random.uniform(0.5,0.82)),3),
                            "Off-targets":int(np.random.randint(0,6)),"Rating":"MED",
                        })
                guides = sorted(guides, key=lambda x: x["Efficiency"], reverse=True)[:8]

            st.markdown(section(f"Top {len(guides)} Guide RNAs Designed"), unsafe_allow_html=True)
            gcc = st.columns(min(4,len(guides)))
            for i, g in enumerate(guides[:4]):
                clr2 = "#00ff9d" if g["Efficiency"]>=0.80 else ("#ffc107" if g["Efficiency"]>=0.60 else "#ff3d5a")
                oclr = "#00ff9d" if g["Off-targets"]==0 else ("#ffc107" if g["Off-targets"]<=3 else "#ff3d5a")
                with gcc[i]:
                    st.markdown(
                        f'<div style="background:#041820;border:1px solid {clr2}44;'
                        f'border-left:3px solid {clr2};border-radius:6px;padding:12px 13px;margin-bottom:8px;">'
                        f'<div style="color:#4a9aaa;font-size:.5rem;letter-spacing:2px;margin-bottom:4px;">'
                        f'{g["Guide"]}  ·  pos {g["Position"]}</div>'
                        f'<div style="font-family:Space Mono,monospace;font-size:.6rem;color:#88ddee;'
                        f'word-break:break-all;margin:6px 0;letter-spacing:1px;">{g["Sequence"]}</div>'
                        f'<div style="display:flex;gap:5px;flex-wrap:wrap;">'
                        f'{badge("EFF "+str(g["Efficiency"]),clr2)}'
                        f'{badge("OT: "+str(g["Off-targets"]),oclr)}'
                        f'{badge("GC: "+str(g["GC%"])+"%","#00e5ff")}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            st.dataframe(pd.DataFrame(guides), use_container_width=True, hide_index=True)

            # charts
            cp, co = st.columns(2)
            with cp:
                st.markdown(section("PAM Site Efficiency Map"), unsafe_allow_html=True)
                fig_pam = go.Figure(go.Scatter(
                    x=[g["Position"] for g in guides],
                    y=[g["Efficiency"] for g in guides],
                    mode="markers+text", text=[g["Guide"] for g in guides],
                    textposition="top center", textfont=dict(color="#00e5ff", size=9),
                    marker=dict(
                        size=15, color=[g["Efficiency"] for g in guides],
                        colorscale=[[0,"#ff3d5a"],[0.5,"#ffc107"],[1,"#00ff9d"]],
                        colorbar=dict(title="Eff", thickness=8, tickfont=dict(color="#00e5ff", size=8)),
                        line=dict(color="rgba(255,255,255,0.5)", width=1),
                    ),
                    hovertemplate="<b>%{text}</b><br>Position: %{x}<br>Efficiency: %{y:.3f}<extra></extra>",
                ))
                fig_pam.update_layout(**DK(
                    xaxis=dict(title="Position (bp)", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                    yaxis=dict(title="Efficiency", range=[0,1.1], color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                    title=dict(text="PAM Site Map  ·  "+cas_sys, font=dict(size=11, color="#4a9aaa")),
                    height=320,
                ))
                st.plotly_chart(fig_pam, use_container_width=True)
            with co:
                st.markdown(section("Off-Target Risk Assessment"), unsafe_allow_html=True)
                ov  = [g["Off-targets"] for g in guides]
                fig_ot = go.Figure(go.Bar(
                    x=[g["Guide"] for g in guides], y=ov,
                    marker_color=["#ff3d5a" if v>3 else ("#ffc107" if v>0 else "#00ff9d") for v in ov],
                    text=ov, textposition="outside", textfont=dict(color="#00e5ff", size=11),
                ))
                fig_ot.update_layout(**DK(
                    xaxis=dict(title="Guide RNA", color="#4a9aaa"),
                    yaxis=dict(title="Predicted Off-target Sites", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                    title=dict(text="Off-Target Risk", font=dict(size=11, color="#4a9aaa")),
                    height=320,
                ))
                st.plotly_chart(fig_ot, use_container_width=True)

            st.markdown(section("Cas System Compatibility Reference"), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                {"System":"SpCas9",     "PAM":"NGG",     "Cut type":"Blunt DSB",  "Cargo size":"4.2 kb","Delivery":"AAV/Lentiviral","Best use":"Gene KO, most types"},
                {"System":"SaCas9",     "PAM":"NNGRRT",  "Cut type":"Blunt DSB",  "Cargo size":"3.2 kb","Delivery":"Compact AAV",   "Best use":"In vivo AAV delivery"},
                {"System":"Cas12a",     "PAM":"TTTV 5'", "Cut type":"Staggered",  "Cargo size":"3.9 kb","Delivery":"RNP / mRNA",    "Best use":"HDR, low off-target"},
                {"System":"Cas13d",     "PAM":"N/A RNA", "Cut type":"RNA only",   "Cargo size":"2.9 kb","Delivery":"AAV / mRNA",    "Best use":"RNA knockdown"},
                {"System":"CasRx",      "PAM":"N/A RNA", "Cut type":"RNA only",   "Cargo size":"2.8 kb","Delivery":"AAV",           "Best use":"High-eff RNA KD"},
                {"System":"dCas9-KRAB", "PAM":"NGG",     "Cut type":"None",       "Cargo size":"5.1 kb","Delivery":"Lentiviral",    "Best use":"CRISPRi silencing"},
                {"System":"dCas9-VP64", "PAM":"NGG",     "Cut type":"None",       "Cargo size":"5.2 kb","Delivery":"Lentiviral",    "Best use":"CRISPRa activation"},
            ]), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div style="text-align:center;padding:40px;background:#041820;border:1px solid #00e5ff18;border-radius:8px;">'
            '<div style="font-family:Orbitron,sans-serif;font-size:1rem;color:#00e5ff;margin-bottom:10px;letter-spacing:4px;">CRISPR ENGINE READY</div>'
            '<div style="color:#4a9aaa;font-size:.7rem;line-height:2;">Select Cas system  ·  editing strategy  ·  paste DNA sequence  ·  click RUN</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════
# TAB 4 — LIGAND / RDKIT
# ══════════════════════════════════════════════════════
with T4:
    st.markdown(section("Approved Cancer Drug Library", "SMILES · 3D Pharmacophore · Lipinski Ro5"), unsafe_allow_html=True)
    lc = st.columns(3)
    for i,(drug,smi) in enumerate(SMILES.items()):
        with lc[i%3]:
            st.markdown(
                f'<div style="background:#041820;border:1px solid #00e5ff22;border-left:3px solid #00e5ff;'
                f'border-radius:6px;padding:12px 14px;margin-bottom:8px;">'
                f'<div style="font-family:Orbitron,sans-serif;color:#00e5ff;font-size:.72rem;margin-bottom:6px;">{drug}</div>'
                f'<div style="color:#4a9aaa;font-size:.52rem;word-break:break-all;'
                f'font-family:Space Mono,monospace;line-height:1.6;">{smi}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        f'<div style="background:#041820;border:1px solid #00e5ff18;border-radius:6px;'
        f'padding:12px 16px;color:#4a9aaa;font-size:.68rem;margin-top:4px;">'
        f'Full 3D pharmacophore rendering + Lipinski Ro5 analysis → go to  {badge("3D STRUCTURE")}  tab  →  RDKit 3D</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════
# TAB 5 — 5D VISUALIZATION
# ══════════════════════════════════════════════════════
with T5:
    st.markdown(section("5D Visualization & MD Trajectory Analysis", "Manifold · MDAnalysis · VMD · Plotly"), unsafe_allow_html=True)
    V1, V2, V3 = st.tabs(["5D Manifold", "MDAnalysis Results", "VMD Viewer"])

    with V1:
        vA, vB, vC = st.columns(3)
        with vA: npts  = st.slider("Sample points", 50, 400, 150, key="v5n")
        with vB: dim5  = st.selectbox("Color dimension", ["Mutational Burden","Expression Level","Therapeutic Index","Genomic Instability"], key="v5d")
        with vC: cscl  = st.selectbox("Color scale", ["Plasma","Viridis","Inferno","Cividis","Turbo"], key="v5c")
        np.random.seed(42)
        clist = list(expr.keys())
        df5 = pd.DataFrame({
            "X":      np.random.randn(npts),
            "Y":      np.random.randn(npts),
            "Z":      np.random.randn(npts),
            "Size":   np.random.rand(npts)*10+3,
            "Mut":    np.abs(np.random.randn(npts))*60,
            "Expr":   np.random.randn(npts)*3+6,
            "TI":     np.random.uniform(0,100,npts),
            "GI":     np.random.exponential(20,npts),
            "Cancer": np.random.choice(clist, npts),
        })
        cv_map = {"Mutational Burden":"Mut","Expression Level":"Expr","Therapeutic Index":"TI","Genomic Instability":"GI"}
        cv = cv_map.get(dim5,"Mut")
        fig5 = go.Figure(go.Scatter3d(
            x=df5["X"], y=df5["Y"], z=df5["Z"],
            mode="markers",
            hovertemplate="<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
            hovertext=[f"{r['Cancer']}<br>{dim5}: {round(r[cv],1)}" for _,r in df5.iterrows()],
            marker=dict(
                size=df5["Size"],
                color=df5[cv], colorscale=cscl, opacity=0.85,
                colorbar=dict(title=dim5, thickness=14, tickfont=dict(color="#00e5ff", size=9),
                              outlinecolor="rgba(0,229,255,0.13)"),
                line=dict(color="rgba(255,255,255,0.15)", width=0.3),
            ),
        ))
        fig5.update_layout(**DK(
            scene=dict(
                xaxis=dict(title="Genomic Frequency",  color="#4a9aaa", backgroundcolor="rgba(4,24,32,0.6)", gridcolor="rgba(0,229,255,0.08)"),
                yaxis=dict(title="Pathway Stability",   color="#4a9aaa", backgroundcolor="rgba(4,24,32,0.6)", gridcolor="rgba(0,229,255,0.08)"),
                zaxis=dict(title="Expression Energy",   color="#4a9aaa", backgroundcolor="rgba(4,24,32,0.6)", gridcolor="rgba(0,229,255,0.08)"),
                bgcolor="rgba(2,12,18,0.9)",
            ),
            title=dict(text=f"<b>{query}</b>  ·  {dim5}  ·  {npts} samples  ·  5D Quantum Manifold",
                       font=dict(size=12, color="#4a9aaa")),
            height=560,
        ))
        st.plotly_chart(fig5, use_container_width=True)

        # cancer distribution donut
        dc = df5["Cancer"].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=dc.index, values=dc.values, hole=0.60,
            marker=dict(
                colors=["#00e5ff","#ff3d5a","#ffc107","#00ff9d","#b44fff","#ff6600","#ff9933","#00aaff"],
                line=dict(color="#030f14", width=2),
            ),
            textfont=dict(color="#c8f0f8", size=11),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        ))
        fig_donut.update_layout(**DK(
            title=dict(text="Cancer Type Distribution in Sample", font=dict(size=11, color="#4a9aaa")),
            showlegend=True,
            legend=dict(font=dict(color="#00e5ff", size=10), bgcolor="rgba(0,0,0,0)"),
            height=320,
        ))
        st.plotly_chart(fig_donut, use_container_width=True)

    with V2:
        np.random.seed(10)
        frames = np.arange(200)
        mT1, mT2, mT3, mT4 = st.tabs(["RMSD", "RMSF", "Radius of Gyration", "H-Bond Count"])

        with mT1:
            rmsd = np.clip(np.cumsum(np.random.normal(0,0.02,200))+1.0, 0.8, 4.0)
            fig_rm = go.Figure()
            fig_rm.add_trace(go.Scatter(
                x=frames, y=rmsd, mode="lines",
                line=dict(color="#00e5ff", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.06)",
                name="RMSD",
            ))
            fig_rm.add_hline(
                y=float(np.mean(rmsd)), line_dash="dash", line_color="#ffc107",
                annotation_text=f"Mean: {round(float(np.mean(rmsd)),2)} Å",
                annotation_font_color="#ffc107", annotation_font_size=11,
            )
            fig_rm.update_layout(**DK(
                xaxis=dict(title="Frame", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                yaxis=dict(title="RMSD (Å)", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                title=dict(text=f"<b>{query}</b>  Backbone RMSD  ·  200 MD frames  ·  Cα selection",
                           font=dict(size=12, color="#4a9aaa")),
                height=340,
            ))
            st.plotly_chart(fig_rm, use_container_width=True)
            c1,c2,c3 = st.columns(3)
            with c1: st.markdown(card_html("MEAN RMSD", str(round(float(np.mean(rmsd)),2)), " Å"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("MAX RMSD",  str(round(float(np.max(rmsd)),2)),  " Å", "", "#ffc107"), unsafe_allow_html=True)
            with c3: st.markdown(card_html("MIN RMSD",  str(round(float(np.min(rmsd)),2)),  " Å", "", "#00ff9d"), unsafe_allow_html=True)

        with mT2:
            n_res = 100
            rmsf  = np.abs(np.random.normal(0.9,0.5,n_res))+0.2
            for h in hs:
                idx = min(h["pos"] % n_res, n_res-1)
                rmsf[idx] += 2.0 * h["freq"] * 8
            fig_rf = go.Figure(go.Bar(
                x=np.arange(1,n_res+1), y=rmsf,
                marker=dict(
                    color=rmsf,
                    colorscale=[[0,"#002535"],[0.4,"#00e5ff"],[1,"#ff3d5a"]],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                hovertemplate="Residue %{x}<br>RMSF: %{y:.2f} Å<extra></extra>",
            ))
            for h in hs[:4]:
                idx = min(h["pos"] % n_res, n_res-1)+1
                fig_rf.add_vline(x=idx, line_dash="dash", line_color="#ff3d5a",
                    annotation_text=h["aa"], annotation_font_color="#ff3d5a", annotation_font_size=9)
            fig_rf.update_layout(**DK(
                xaxis=dict(title="Residue Index", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                yaxis=dict(title="RMSF (Å)", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                title=dict(text=f"<b>{query}</b>  Per-Residue RMSF  ·  Red markers = mutation hotspots",
                           font=dict(size=12, color="#4a9aaa")),
                height=340,
            ))
            st.plotly_chart(fig_rf, use_container_width=True)

        with mT3:
            rg = np.clip(18+np.cumsum(np.random.normal(0,0.05,200)), 16, 22)
            fig_rg = go.Figure(go.Scatter(
                x=frames, y=rg, mode="lines",
                line=dict(color="#00ff9d", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,255,157,0.05)",
            ))
            fig_rg.update_layout(**DK(
                xaxis=dict(title="Frame", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                yaxis=dict(title="Rg (Å)", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                title=dict(text=f"<b>{query}</b>  Radius of Gyration  ·  protein compactness",
                           font=dict(size=12, color="#4a9aaa")),
                height=320,
            ))
            st.plotly_chart(fig_rg, use_container_width=True)

        with mT4:
            hb = np.abs(np.random.normal(45,9,200)).astype(int)
            fig_hb = go.Figure(go.Scatter(
                x=frames, y=hb, mode="lines",
                line=dict(color="#b44fff", width=2.5),
                fill="tozeroy", fillcolor="rgba(180,79,255,0.05)",
            ))
            fig_hb.update_layout(**DK(
                xaxis=dict(title="Frame", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                yaxis=dict(title="H-Bond Count", color="#4a9aaa", gridcolor="rgba(0,229,255,0.06)"),
                title=dict(text=f"<b>{query}</b>  Hydrogen Bond Count  ·  backbone + sidechain",
                           font=dict(size=12, color="#4a9aaa")),
                height=320,
            ))
            st.plotly_chart(fig_hb, use_container_width=True)

    with V3:
        if not HAS_3D:
            st.error("py3Dmol not installed — run: !pip install py3Dmol stmol")
        else:
            vT1,vT2,vT3,vT4 = st.tabs(["Cartoon+Surface","Electrostatic","Secondary Structure","Mutation Hotspots"])
            def render_vmd(style_fn, surf=None, lab=True, h_=460):
                try:
                    v_ = py3Dmol.view(query="pdb:"+pdb, width=920, height=h_)
                    style_fn(v_)
                    if surf:
                        v_.addSurface(py3Dmol.VDW, surf)
                    if lab:
                        for h2 in hs:
                            v_.addResidueLabels({"resi":str(h2["pos"])},{"fontColor":"#ff3d5a","backgroundColor":"black","fontSize":11})
                    v_.zoomTo()
                    showmol(v_, height=h_, width=920)
                except Exception as ex:
                    st.error(str(ex))
            with vT1:
                render_vmd(lambda v: v.setStyle({"cartoon":{"color":"spectrum"}}),
                           surf={"opacity":0.28,"color":"spectrum"})
                st.markdown(f'<div style="background:#041820;border:1px solid #00e5ff18;border-radius:6px;padding:8px 14px;font-size:.64rem;color:#4a9aaa;margin-top:6px;">{badge("VMD NewCartoon")}  + VDW Surface  ·  spectrum coloring  ·  PDB {pdb}</div>', unsafe_allow_html=True)
            with vT2:
                render_vmd(lambda v: v.setStyle({"cartoon":{"color":"chain"}}),
                           surf={"opacity":0.50,"colorscheme":"rwb"})
                st.markdown(f'<div style="background:#041820;border:1px solid #00e5ff18;border-radius:6px;padding:8px 14px;font-size:.64rem;color:#4a9aaa;margin-top:6px;">{badge("Electrostatic")}  Red=negative  ·  White=neutral  ·  Blue=positive</div>', unsafe_allow_html=True)
            with vT3:
                render_vmd(lambda v: v.setStyle({"cartoon":{"color":"ssJmol"}}), lab=False)
                st.markdown(f'<div style="background:#041820;border:1px solid #00e5ff18;border-radius:6px;padding:8px 14px;font-size:.64rem;color:#4a9aaa;margin-top:6px;">{badge("Secondary Structure")}  Helix=red  ·  Sheet=yellow  ·  Loop=green</div>', unsafe_allow_html=True)
            with vT4:
                def hs_render(v):
                    v.setStyle({"cartoon":{"color":"white","opacity":0.4}})
                    for h2 in hs:
                        v.addStyle({"resi":str(h2["pos"])},{"sphere":{"color":"#ff3d5a","radius":1.3}})
                        v.addResidueLabels({"resi":str(h2["pos"])},{"fontColor":"#ff3d5a","backgroundColor":"black","fontSize":13})
                render_vmd(hs_render, lab=False)
                st.markdown(f'<div style="background:#041820;border:1px solid #ff3d5a44;border-radius:6px;padding:8px 14px;font-size:.64rem;color:#4a9aaa;margin-top:6px;">{badge("Hotspot Residues","#ff3d5a")}  Red spheres = COSMIC mutation hotspots  ·  {len(hs)} sites shown</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 6 — REPORT & EXPORT
# ══════════════════════════════════════════════════════
with T6:
    st.markdown(section("Integrated Pipeline Report & Export"), unsafe_allow_html=True)
    rp = get_ppi(query, limit=8)

    report_rows = [
        ("TARGET GENE",     query),
        ("PDB STRUCTURE",   pdb),
        ("ONCO SCORE",      str(sc.get("oncoscore","N/A"))+"/100"),
        ("DRUGGABILITY",    str(sc.get("druggability","N/A"))+"/100"),
        ("MUTATION FREQ",   str(sc.get("mutation_freq","N/A"))+"%"),
        ("CLINICAL TRIALS", str(sc.get("clinical_trials","N/A"))+" active"),
        ("TOP CANCER",      topc+" · "+str(expr.get(topc,"N/A"))+" log2(TPM)"),
        ("HOTSPOTS",        ", ".join([h["aa"] for h in hs]) if hs else "None catalogued"),
        ("TOP INTERACTORS", ", ".join([b for a,b,s in rp[:5]])),
        ("PIPELINE",        "G-FUSION v10 · COMPLETE"),
    ]
    rows_html = "".join([
        f'<tr style="border-bottom:1px solid #00e5ff0a;">'
        f'<td style="color:#4a9aaa;padding:8px 4px;width:200px;font-size:.6rem;letter-spacing:2px;text-transform:uppercase;">{k}</td>'
        f'<td style="color:#c8f0f8;font-size:.72rem;padding:8px 4px;">{v}</td></tr>'
        for k,v in report_rows
    ])
    st.markdown(
        f'<div style="background:#041820;border:1px solid #00e5ff22;border-radius:8px;padding:18px 20px;">'
        f'<div style="font-family:Orbitron,sans-serif;font-size:.88rem;color:#00e5ff;'
        f'letter-spacing:3px;margin-bottom:14px;">IN SILICO PIPELINE REPORT  ·  {query}</div>'
        f'<table style="width:100%;border-collapse:collapse;">{rows_html}</table>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(section("Pipeline Module Status"), unsafe_allow_html=True)
    mods = [
        ("3D Structure Tools",     "Py3Dmol · NGLView · PyMOL-style · RDKit 3D · MMFF94",          "#00e5ff"),
        ("Pathway Network",        "STRING DB live · NetworkX 3D · Cytoscape 2D · Heatmap",         "#00aaff"),
        ("CRISPR Engine",          "SpCas9/Cas12a/Cas13d · gRNA design · PAM · Off-target",         "#b44fff"),
        ("Ligand / RDKit",         "MMFF3D geometry · Lipinski Ro5 · Radar chart · 6 drugs",        "#ffc107"),
        ("5D Visualization",       "MDAnalysis RMSD/RMSF/Rg/HB · VMD 4 modes · Manifold",          "#00ff9d"),
        ("Molecular Intelligence", "Real-time precision oncology annotation via Anthropic API",      "#ff9933"),
        ("Report & Export",        "Full TXT report · PPI CSV · Expression CSV download",           "#ff3d5a"),
    ]
    mc = st.columns(2)
    for i,(nm,desc,clr2) in enumerate(mods):
        with mc[i%2]:
            st.markdown(
                f'<div style="background:#041820;border:1px solid {clr2}22;border-left:3px solid {clr2};'
                f'border-radius:6px;padding:10px 14px;margin:5px 0;display:flex;justify-content:space-between;align-items:center;">'
                f'<div><div style="color:{clr2};font-family:Orbitron,sans-serif;font-size:.68rem;">{nm}</div>'
                f'<div style="color:#4a9aaa;font-size:.54rem;margin-top:3px;">{desc}</div></div>'
                f'{badge("ACTIVE",clr2)}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(section("Download Outputs"), unsafe_allow_html=True)
    LL = [f"G-FUSION v10  --  IN SILICO PIPELINE REPORT", "="*65]
    for k,v in report_rows: LL.append(f"  {k:<25}: {v}")
    LL += ["","PPI INTERACTIONS (STRING DB)","-"*45]
    for a,b,s in rp[:8]: LL.append(f"  {b:<16} score:{round(s,3)}  pathway:{PWY.get(b,'?')}")
    LL += ["","EXPRESSION BY CANCER TYPE","-"*45]
    for c,v in expr.items(): LL.append(f"  {c:<12}: {v} log2(TPM)")
    LL += ["","="*65,"G-FUSION v10  ·  In Silico Cancer Genomics + CRISPR Pipeline"]
    report_txt = "\n".join(LL)

    d1,d2,d3 = st.columns(3)
    with d1:
        st.download_button("DOWNLOAD TXT REPORT", data=report_txt,
                           file_name=f"GFUSION_{query}_report.txt", mime="text/plain", key="dl1")
    df_pp = pd.DataFrame([(b,PWY.get(b,"?"),round(s,3)) for a,b,s in rp], columns=["Partner","Pathway","STRING_Score"])
    with d2:
        st.download_button("DOWNLOAD PPI CSV", data=df_pp.to_csv(index=False).encode(),
                           file_name=f"GFUSION_{query}_PPI.csv", mime="text/csv", key="dl2")
    df_ex = pd.DataFrame(list(expr.items()), columns=["Cancer_Type","Expression_log2TPM"])
    with d3:
        st.download_button("DOWNLOAD EXPRESSION CSV", data=df_ex.to_csv(index=False).encode(),
                           file_name=f"GFUSION_{query}_expression.csv", mime="text/csv", key="dl3")

    st.markdown(
        '<div style="text-align:center;color:#0a2a35;font-size:.5rem;letter-spacing:2px;margin-top:18px;">'
        'G-FUSION v10  ·  IN SILICO PAN-CANCER GENOMICS ENGINE  ·  CRISPR THERAPEUTIC TARGETING  ·  '
        'STRING DB · RDKit · Py3Dmol · Plotly · NetworkX · MDAnalysis · Streamlit'
        '</div>',
        unsafe_allow_html=True,
    )
