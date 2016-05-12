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
       "referer": "http://museum.classics.cam.ac.uk/collections/casts"}
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
        if tag == 'div':
            at = findAttr(attrs,'class')
            if (at is not None) and at[1] == 'views-field views-field-title':
                self.state = 0
                return
        if self.state == 0:
            if tag == 'a':
                at = findAttr(attrs,'href')
                if (at is not None) and at[1].startswith('/collections/casts'):
                    self.imageList.append(at[1])
                    self.state = -1
            return

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
        if self.state < 0:
            if tag == 'div':
                at = findAttr(attrs,'class')
                if (at is not None) and at[1] == 'field field-name-field-picture field-type-image field-label-hidden':
                    self.state = 0
            return
        if self.state == 0:
            if tag == 'img':
                at = findAttr(attrs,'src')
                if at is not None:
                    self.imgUrl = at[1]
                self.state = 1
            return
        if self.state == 1:
            if tag == 'div':
                at = findAttr(attrs,'class')
                if (at is not None) and at[1] == 'field field-name-body field-type-text-with-summary field-label-hidden':
                    self.backup = 1
                    self.state = 4
                    self.acc = 1
                    self.dataBuf = ''
            return
        if self.state == 2:
            if tag == 'div':
                at = findAttr(attrs,'class')
                if (at is not None) and at[1] == 'field-label':
                    self.backup = 2
                    self.state = 4
                    self.acc = 1
                    self.dataBuf = ''
            return
        if self.state == 3:
            if tag == 'div':
                at = findAttr(attrs,'class')
                if (at is not None) and at[1] == 'field-items':
                    self.backup = 3
                    self.state = 4
                    self.acc = 1
                    self.dataBuf = ''
            return
        if self.state == 4:
            if tag == 'div':
                self.acc+=1
    def handle_endtag(self, tag):
        if self.state == 1:
            if tag == 'span':
                self.state = 2
                self.meta['Title'] = self.dataBuf
            return
        if tag == 'div' and self.state == 4:
            self.acc -= 1
            if self.acc <= 0:
                if self.backup == 1:
                    self.meta['Summary'] = self.dataBuf
                    self.state = 2
                if self.backup == 2:
                    self.metaName = self.dataBuf
                    self.state = 3
                if self.backup == 3:
                    self.meta[self.metaName] = self.dataBuf
                    self.state = 2
            return
    def handle_data(self, data):
        tdata = data.strip()
        if self.state == 4:
            if tdata != '':
                self.dataBuf = self.dataBuf + ' ' + tdata
    def sreset(self):
        self.newList = []
        self.state = -1
        self.acc = 0
        self.dataBuf = ''
        self.meta = {}
        self.imgUrl = ''
        self.reset()
    def getResult(self):
        return (self.imgUrl, self.meta, self.newList)
    def work(self,cstr):
        self.sreset()
        self.feed(cstr)


CONTENT_BASE = 'http://museum.classics.cam.ac.uk/collections/casts?title=&field_original_location_value=&field_references_value=&field_provenance_value=&field_number_value=&page='
BEGIN = 1
END = 1
PAGE_BASE = 'http://museum.classics.cam.ac.uk'
BASE_PATH = '../../database/museum.classics.cam.ac.uk/'
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
        pid = url[url.find('casts/')+6:]
        flag,ctstr = fetcher(IMAGE_BASE+imgUrl, pageUrl)
        if flag:
            f = open(BASE_PATH+pid+'.jpg','w')
            f.write(ctstr)
            f.close()
        else:
            print 'passing',IMAGE_BASE+imgUrl
        f = open(BASE_PATH+pid+'.json','w')
        json.dump(meta,f)
        f.close()
        
    for idx in ['']+range(BEGIN,END+1):
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
            workOnUrl(url)
            
                
