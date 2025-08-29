import requests
import xml.etree.ElementTree as ET

def buscar_artigos(term="cardiovascular disease", n=5):
    """
        Busca os artigos na base de dados da PubMed e 
        retorna uma lista com dicionários que representam os artigos sumarizados
    """
    
    # 1. Buscar IDs
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": term, "retmax": n, "retmode": "json"}
    res = requests.get(url, params=params).json()
    ids = res["esearchresult"]["idlist"]

    # 2. Buscar detalhes
    url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
    res = requests.get(url_fetch, params=params)

    root = ET.fromstring(res.text)
    artigos = []
    for artigo in root.findall(".//PubmedArticle"):
        titulo = artigo.findtext(".//ArticleTitle")
        abstract = " ".join([a.text for a in artigo.findall(".//AbstractText") if a.text])
        artigos.append({"titulo": titulo, "abstract": abstract})
    
    return artigos

print(buscar_artigos())
