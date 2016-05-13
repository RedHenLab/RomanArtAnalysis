# -*- coding: utf-8 -*-
#downloader for some single pages

from HTMLParser import HTMLParser
import urllib2
import urllib
import time
import socket
import simplejson as json
import sys,os

socket.setdefaulttimeout(10)
ua = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0',\
       "referer": "http://www.romancoins.info/"}
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
        
def findAttr(att,tar):
    for at in att:
        if at[0] == tar:
            return at
    return None


class pageParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.sreset()
    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            at = findAttr(attrs,'src')
            if at is not None:
                if at[1] not in self.imgHash:
                    self.imgUrl.append(at[1])
                    self.imgName.append(at[1].replace('/','_'))
                    att = findAttr(attrs,'alt')
                    if att is not None:
                        self.meta[self.imgName[-1]] = att[1]
                    self.imgHash[at[1]] = True
    def handle_endtag(self, tag):
        pass
    def handle_data(self, data):
        tdata = data.strip()
    def sreset(self):
        self.newList = []
        self.state = -1
        self.acc = 0
        self.dataBuf = ''
        self.meta = {}
        self.imgUrl = []
        self.imgName = []
        self.imgHash = {}
        self.reset()
    def getResult(self):
        return (self.imgUrl, self.meta, self.imgName)
    def work(self,cstr):
        self.sreset()
        self.feed(cstr)


BEGIN = 1
END = 1
PAGE_BASE = 'http://www.metmuseum.org/toah/hd/ropo/'
BASE_PATH = '../../database/metmuseum.org/'
IMAGE_BASE = 'http:'
PAGE_LIST = ['hd_ropo.htm']
#PAGE_LIST = ['1b.HTML','2.HTML','3.HTML','3b.HTML','1.HTML','1a1.HTML','1a2.HTML','1a3.HTML']
imagehash = {}
urlPool = []

if __name__ == '__main__':
    if len(sys.argv)>1:
        BASE_PATH = sys.argv[1]
    if len(sys.argv)>2:
        PAGE_BASE = sys.argv[2]
        PAGE_LIST = [sys.argv[3]]
    pp = pageParser()
    def workOnUrl(url):
        global urlPool
        if url in imagehash:
            return
        pageUrl = PAGE_BASE+url
        print 'working on',pageUrl
        imagehash[url] = True
        flag, ctstr = fetcher(pageUrl)
        if not flag:
            print 'passing',url
        ctstr = unicode(ctstr, errors='ignore')
        pp.work(ctstr)
        imgUrl, meta, nameList = pp.getResult()
        meta['page_url'] = pageUrl
        meta['image_url'] = imgUrl
        for iturl,pid in zip(imgUrl,nameList):
            print 'working on',IMAGE_BASE+iturl,'after sleep 1'
            time.sleep(1)
            flag,ctstr = fetcher(IMAGE_BASE+iturl, pageUrl)
            if flag:
                f = open(BASE_PATH+pid+'.jpg','w')
                f.write(ctstr)
                f.close()
            else:
                print 'passing',IMAGE_BASE+iturl
        f = open(BASE_PATH+url+'.json','w')
        json.dump(meta,f)
        f.close()
        
    for idx in PAGE_LIST:
        workOnUrl(idx)
            
                
