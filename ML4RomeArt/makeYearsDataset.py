# -*- coding: utf-8 -*-
import sys,os
import simplejson as json
import glob

if __name__ == '__main__':
    assert len(sys.argv) > 2
    imgDir = sys.argv[1]
    txtDir = sys.argv[2]
    outPut = sys.argv[3]
    OF = open(outPut,'w')
    imgList = glob.glob(os.path.join(imgDir,'*.jpg'))
    maxy = -10000
    miny = 10000
    allY = []
    pair = []
    for img in imgList:
        name = os.path.split(img)[1].split('.')[0]
        try:
            ss = open(os.path.join(txtDir,name+'.txt'),'r').read()
        except:
            print 'fail no txt',name
            continue
        ss = ss.strip()
        ss = ss.replace('=',':')
        ss = ss.replace(' ','')
        ss = ss.replace('“','"')
        ss = ss.replace('”','"')
        if ss[0] != '{':
            ss = '{'+ss
        ss = ss.replace('/','-')
        ss = ss.replace(' ','')
        if ss.find('Keywords') != -1:
            ss = ss[:ss.find('Keywords')]
        adPos = ss.rfind('AD')
        neg = 1
        if adPos == -1:
            adPos = ss.rfind('BC')
            neg = -1
        if adPos == -1:
            print name,'fail no year'
            continue
        vs = ss[:adPos]
        i = adPos - 1
        sts = 0
        ts = ''
        st = []
        while i>=0:
            if sts == 0:
                if vs[i].isdigit():
                    ts = vs[i]+ts
                if (not vs[i].isdigit()) and (vs[i] != '-'):
                    if len(ts) > 0:
                        st.append(int(ts)*neg)
                    break
                if vs[i] == '-':
                    if len(ts) > 0:
                        st.append(int(ts)*neg)
                    else:
                        break
                    sts = 1
                i-=1
                continue
            if sts == 1:
                ts  = ''
                if vs[i-1:i+1] == 'AD' or vs[i-1:i+1] == 'BC':
                    sts = 2
                    if vs[i-1:i+1] == 'BC':
                        neg = -1
                    i-=2
                    continue
                if vs[i].isdigit():
                    sts = 2
                    continue
                break
            if sts == 2:
                if vs[i].isdigit():
                    ts = vs[i]+ts
                    i-=1
                    continue
                if len(ts) > 0:
                    st.append(int(ts)*neg)
                break
        if len(st) < 1:
            print name,'fail parse year'
            continue
        checkFlag = False
        for y in st:
            if abs(y) > 1000:
                print 'fail with year check',y
                checkFlag = True
        if len(st) > 1 and st[1] > st[0]:
            print 'fail with year check',st
            continue
        print name,st

        if len(st) > 1:
            st[0] = (st[0]+st[1])/2
        maxy = max(maxy,st[0])
        miny = min(miny,st[0])
        pair.append((img,st[0]))
        allY.append(st[0])
        print >>OF,img,st[0]
    print 'min,max',miny,maxy
    allY.sort()
    t1 = allY[len(allY)/3]
    t2 = allY[len(allY)/3*2]
    print t1,t2
    OF.close()
    OF = open(outPut+'_class.txt','w')
    for p in pair:
        print >>OF,p[0],
        if p[1] < t1:
            tag = 0
        elif p[1] < t2:
            tag = 1
        else:
            tag = 2
        print >>OF,tag
            
            
                

                
        
        
        

