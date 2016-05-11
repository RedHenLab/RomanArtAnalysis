# -*- coding: utf-8 -*-
#downloader for http://ancientrome.ru/

from HTMLParser import HTMLParser
import urllib2
import urllib
import time
import socket
import simplejson as json

socket.setdefaulttimeout(5)
ua = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0',\
       "referer": "http://ancientrome.ru/"}
def fetcher(url, ref = None):
    if ref is not None:
        ua['referer'] = ref
    req = urllib2.Request(url, None, ua)
    try:
        soc = urllib2.urlopen(req,timeout=5)
        cont = soc.read()
        if len(cont)<1000:
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
            at = findAttr(attrs,'href')
            if (at is not None) and at[1].startswith('img.htm?id='):
                self.imageList.append(at[1])
    def handle_endtag(self, tag):
        pass
    def handle_data(self, data):
        pass
    def sreset(self):
        self.imageList = []
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
            if (at is not None) and at[1].startswith('img.htm?id='):
                self.newList.append(at[1])
        if self.state < 0:
            if tag == 'noscript':
                self.state = 0
                return
        if self.state == 0:
            if tag == 'a':
                at = findAttr(attrs,'href')
                if at is not None:
                    self.imgUrl = at[1]
                    self.state = 1
            if tag == 'img':
                at = findAttr(attrs,'src')
                if at is not None:
                    self.imgUrl = at[1]
                    self.state = 1
            return
        if self.state == 1:
            if len(tag) == 2 and tag[0]=='h' and tag[1].isdigit():
                self.state = 2
            if tag == 'div':
                at = findAttr(attrs,'class')
                if at is not None:
                    if at[1] == 'label':
                        self.metaName = at[1]
                        self.backup = self.state
                        self.state = 4
                        self.acc = 1
                        self.dataBuf = ''
                    if at[1] == 'detail':
                        self.state = 3
            return
        if self.state == 3:
            if tag == 'div':
                at = findAttr(attrs,'class')
                if at is not None:
                    if at[1].startswith('desct'):
                        self.state = 5
            return
        if self.state == 6:
            if tag == 'div':
                at = findAttr(attrs,'class')
                if at is not None:
                    if at[1] == 'desc':
                        self.backup = 3
                        self.state = 4
                        self.acc = 1
                        self.dataBuf = ''
            return
        if self.state == 4:
            if tag == 'div':
                self.acc+=1
            if tag in ignoreData:
                self.state = 7
                self.last = tag
    def handle_endtag(self, tag):
        if self.state == 2:
            self.state = 1
            return
        if tag == 'div' and self.state == 4:
            self.acc -= 1
            if self.acc <= 0:
                self.meta[self.metaName] = self.dataBuf
                self.state = self.backup
            return
        if self.state == 5:
            self.state = 6
            return
        if self.state == 7:
            if tag == self.last:
                self.state = 4
            return
    def handle_data(self, data):
        tdata = data.strip()
        if self.state == 2:
            if 'title' in self.meta:
                self.meta['Title'].append(tdata)
            else:
                self.meta['Title'] = [tdata]
        if self.state == 4:
            if tdata != '':
                self.dataBuf = self.dataBuf + ' ' + tdata
        if self.state == 5:
            self.metaName = tdata
    def sreset(self):
        self.newList = []
        self.state = -1
        self.acc = 0
        self.dataBuf = ''
        self.meta = {}
        self.reset()
    def getResult(self):
        return (self.imgUrl, self.meta, self.newList)
    def work(self,cstr):
        self.sreset()
        self.feed(cstr)


CONTENT_BASE = 'http://ancientrome.ru/art/artworken/index.htm?id=739&pn=20&sp='
BEGIN = 1
END = 2
PAGE_BASE = 'http://ancientrome.ru/art/artworken/'
BASE_PATH = '../database/arr/'
imagehash = {}
urlPool = []

if __name__ == '__main__':
    cp = contentParser()
    pp = pageParser()
    def workOnUrl(url):
        global urlPool
        if url in imagehash:
            return
        pageUrl = PAGE_BASE+url
        print 'working on',pageUrl,'after sleep 5'
        time.sleep(5)
        imagehash[url] = True
        flag, ctstr = fetcher(pageUrl, cturl)
        if not flag:
            print 'passing',url
        ctstr = unicode(ctstr, errors='ignore')
        pp.work(ctstr)
        imgUrl, meta, newList = pp.getResult()
        urlPool += newList
        meta['page_url'] = pageUrl
        meta['image_url'] = imgUrl
        flag,ctstr = fetcher(imgUrl, pageUrl)
        pid = url[url.find('id=')+3:]
        if flag:
            f = open(BASE_PATH+pid+'.jpg','w')
            f.write(ctstr)
            f.close()
            f = open(BASE_PATH+pid+'.json','w')
            json.dump(meta,f)
            f.close()
        else:
            print 'passing',url
    for idx in xrange(BEGIN,END+1):
        cturl = CONTENT_BASE+str(idx)
        print 'working on',cturl,'after sleep 10'
        if idx!=BEGIN:
            time.sleep(10)
        flag,ctstr = fetcher(cturl)
        if not flag:
            print 'passing',idx
            continue
        ctstr = unicode(ctstr, errors='ignore')
        cp.work(ctstr)
        urlList = cp.getResult()
        for url in urlList:
            #if url != 'img.htm?id=413':
            #    continue
            workOnUrl(url)
        for url in urlPool:
            workOnUrl(url)
            
                
