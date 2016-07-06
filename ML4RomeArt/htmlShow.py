# -*- coding: utf-8 -*-
import sys,os

htmlAllHead = '<html> <body> <table id="tfhover" class="tftable" border="1">'
html1 = '<tr><th> %s </th><th><img src="%s" width="320" height="320" /></th></tr>'
html2 = '<tr><td><img src="%s" /></td><td><img src="%s"/></td></tr>'
htmlEnd = '</table> </body> </html>'

subDir = 'pre-process/'
goodList = 'good_list.txt'
suf = '_us10k__Attributes.png'
showDir = 'show4us10k'

if __name__ == '__main__':
    assert len(sys.argv) >= 3
    rootDir = sys.argv[1]
    outputFile = sys.argv[2]
    lines = open(os.path.join(rootDir,subDir,goodList), 'r').readlines()
    OF = open(outputFile, 'w')
    print >>OF, htmlAllHead
    for line in lines:
        l = line.strip()
        print >>OF, html1 % (l, os.path.join('./',subDir,l+'_frontal.jpg'))
        print >>OF, html2 % (os.path.join('./',l), os.path.join('./',showDir,l+suf))
    print>>OF, htmlEnd
    OF.close()
    
