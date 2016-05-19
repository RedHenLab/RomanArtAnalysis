# -*- coding: utf-8 -*-
#downloader for http://www.rome101.com/portraiture/

from HTMLParser import HTMLParser
import urllib2
import urllib
import time
import socket
import simplejson as json
import sys,os

socket.setdefaulttimeout(10)
ua = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0',\
       "referer": "http://www.rome101.com/portraiture/"}
def fetcher(url, ref = None):
    if ref is not None:
        ua['referer'] = ref
    req = urllib2.Request(url, None, ua)
    try:
        soc = urllib2.urlopen(req,timeout=5)
        cont = soc.read()
        if len(cont) < 1000:
            return False,''
    except urllib2.HTTPError, e:
        print e.code
        return False,''
    except:
        return False,''
    return True,cont
        
ignoreData = ['script','a','span']
def findAttr(att,tar):
    for at in att:
        if at[0] == tar:
            return at
    return None

class contentParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.sreset()
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            at = findAttr(attrs,'style')
            if (at is not None) and at[1] == 'text-decoration: none':
                at = findAttr(attrs,'href')
                if at is not None:
                    self.imageList.append(at[1])

    def handle_endtag(self, tag):
        pass
    def handle_data(self, data):
        pass
    def sreset(self):
        self.imageList = []
        self.state = -1
        self.reset()
    def getResult(self):
        return self.imageList
    def work(self,cstr):
        self.reset()
        self.feed(cstr)


class pageParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.sreset()
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            at = findAttr(attrs,'href')
            if (at is not None) and (at[1].endswith('.html') or at[1].endswith('.htm')):
                self.imgUrl.append(at[1])

    def handle_endtag(self, tag):
        pass
    def handle_data(self, data):
        pass
    def sreset(self):
        self.newList = []
        self.state = -1
        self.acc = 0
        self.dataBuf = ''
        self.meta = {}
        self.imgUrl = []
        self.reset()
    def getResult(self):
        return (self.imgUrl, self.meta, self.newList)
    def work(self,cstr):
        self.sreset()
        self.feed(cstr)


CONTENT_BASE = 'http://www.rome101.com/portraiture/'
BEGIN = 1
END = 1
PAGE_BASE = 'http://www.rome101.com/portraiture/'
BASE_PATH = '../../database/rome101.com/'
IMAGE_BASE = ''
imagehash = {}
urlPool = []

if __name__ == '__main__':
    if len(sys.argv)>1:
        BASE_PATH = sys.argv[1]
    if len(sys.argv)>2:
        BEGIN = int(sys.argv[2])
        END = int(sys.argv[3])
    cp = contentParser()
    pp = pageParser()
    def workOnUrl(url):
        global urlPool
        pageUrl = PAGE_BASE+url
        print 'working on',pageUrl,'after sleep 10'
        #time.sleep(10)
        flag, ctstr = fetcher(pageUrl)
        if not flag:
            print 'passing',url
        ctstr = unicode(ctstr, errors='ignore')
        pp.work(ctstr)
        imgUrl, meta, newList = pp.getResult()
        urlPool += newList
        meta['page_url'] = pageUrl
        meta['image_url'] = imgUrl
        pid = url[:-1]
        IMAGE_BASE = pageUrl
        for turl in imgUrl:
            imPage = IMAGE_BASE+turl
            print 'working on',imPage,'after sleep 5'
            time.sleep(5)
            imPageH,imPageT = os.path.split(imPage)
            flag, ctstr = fetcher(imPage, pageUrl)
            if not flag:
                print 'passing',imPage
                continue
            ctstr = unicode(ctstr, errors='ignore')
            oc  = ctstr.find('pix/')
            ob, oe = ctstr.rfind('"',0,oc), ctstr.find('"',oc)
            if oe-ob-1>0:
                im = ctstr[ob+1:oe]
            else:
                print 'passing',imPage
                continue
            imName = im[im.find('pix/')+4:]
            flag,ctstr = fetcher(imPageH+'/'+im, pageUrl)
            if flag:
                f = open(BASE_PATH+pid+'_'+imName+'.jpg','w')
                f.write(ctstr)
                f.close()
            else:
                print 'passing',IMAGE_BASE+im
        f = open(BASE_PATH+pid+'.json','w')
        json.dump(meta,f)
        f.close()
        
    cturl = CONTENT_BASE
    print 'working on',cturl
    flag,ctstr = fetcher(cturl)
    if not flag:
        print 'passing',cturl
        sys.exit()
    ctstr = unicode(ctstr, errors='ignore')
    cp.work(ctstr)
    urlList = cp.getResult()
    for i,url in enumerate(urlList):
        workOnUrl(url)
            
                
