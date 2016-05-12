# -*- coding: utf-8 -*-
#downloader for http://ancientrome.ru/

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
        
ignoreData = ['script','a','span']
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
        if self.state < 0:
            if tag == 'title':
                self.state = 0
                self.meta['Main'] = []
                self.dataBuf = ''
            return
        if self.state == 1:
            if tag == 'img':
                at = findAttr(attrs,'src')
                if at is not None:
                    self.imgUrl.append(at[1])
                    self.meta['Main'].append(at[1])
            return
    def handle_endtag(self, tag):
        if self.state == 0:
            if tag == 'title':
                self.state = 1
                self.meta['Title'] = self.dataBuf
            return
    def handle_data(self, data):
        tdata = data.strip()
        if self.state == 0:
            if tdata != '':
                self.dataBuf = self.dataBuf + ' ' + tdata
            return
        if self.state == 1:
            self.meta['Main'].append(data)
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


BEGIN = 1
END = 1
PAGE_BASE = 'http://www.romancoins.info/Caesar-Sculpture-'
BASE_PATH = '../../database/romancoins.info/'
IMAGE_BASE = 'http://www.romancoins.info/'
#PAGE_LIST = ['1b.HTML']
PAGE_LIST = ['1b.HTML','2.HTML','3.HTML','3b.HTML','1.HTML','1a1.HTML','1a2.HTML','1a3.HTML']
imagehash = {}
urlPool = []

if __name__ == '__main__':
    if len(sys.argv)>1:
        BASE_PATH = sys.argv[1]
    if len(sys.argv)>2:
        BEGIN = int(sys.argv[2])
        END = int(sys.argv[3])
    pp = pageParser()
    def workOnUrl(url):
        global urlPool
        if url in imagehash:
            return
        pageUrl = PAGE_BASE+url
        print 'working on',pageUrl,'after sleep 5'
        #time.sleep(5)
        imagehash[url] = True
        flag, ctstr = fetcher(pageUrl)
        if not flag:
            print 'passing',url
        ctstr = unicode(ctstr, errors='ignore')
        pp.work(ctstr)
        imgUrl, meta, newList = pp.getResult()
        urlPool += newList
        meta['page_url'] = pageUrl
        meta['image_url'] = imgUrl
        for iturl in imgUrl:
            pid = iturl
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
            
                
