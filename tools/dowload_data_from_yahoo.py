import requests
from config import ROOT_PATH


headers={'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
 'accept-encoding': 'gzip, deflate, br',
 'accept-language': 'en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7',
 'cookie': 'B=7u9trutddg531&b=3&s=f3; uvts=7OU9UhR7f111lJuH; GUC=AQEBAQFbEfNb70IhQwR9&s=AQAAAOWVXffl&g=WxCtng; ucs=lnct=1527819696; HP=1; PRF=t%3D%255EFTSE%253FP%253DFTSE%252B%255EN225%252B%255EFTSE%252BUKXL2.L%252BUKXSP.L%252BUKXM.L%252B%255EAXJO%252BJP%252B%255EGSPC%252B%255EHSI%252B%255EDJI%252BAAPL',
 'referer': 'https://finance.yahoo.com/quote/%5EN225/history?period1=992966400&period2=1529424000&interval=1d&filter=history&frequency=1d',
 'upgrade-insecure-requests': '1',
 'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/66.0.3359.181 Chrome/66.0.3359.181 Safari/537.36'}

def gen_url(url):
    url_f='https://query1.finance.yahoo.com/v7/finance/download/'
    url_b='?period1=992966400&period2=2529424000&interval=1d&events=history&crumb=p.1hS.qBaMS'
    return url_f+url+url_b

urls={
    'jp':gen_url('%5EN225'),
    'bp':gen_url('%5EGSPC'),
    'hs':gen_url('%5EHSI'),
    'uk':gen_url('%5EFTSE'),
    'ax':gen_url('%5EAXJO'),
    'vix':gen_url('%5EVIX')
}

for i in urls:
    r=requests.get(urls[i],headers=headers)
    r.close()
    data=r.text
    with open(ROOT_PATH+'/data/'+i+'.csv','w') as f:
        f.write(data)
        print(i)