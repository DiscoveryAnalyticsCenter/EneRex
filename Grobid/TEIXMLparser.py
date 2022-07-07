import time
from bs4 import BeautifulSoup
import copy
import json
import glob
from pathlib import Path
import argparse
import multiprocessing
from multiprocessing.pool import Pool
import os
from os.path import join
import re
start_time = time.time()

def read_tei(tei_file):
    with open(tei_file, 'r', encoding="utf8") as tei:
        soup = BeautifulSoup(tei, 'lxml-xml')
        return soup
    raise RuntimeError('Cannot generate a soup from the input')


#use it if there's errors with gettext
def elem_to_text(elem, default=''):
    if elem:
        return elem.getText()
    else:
        return default


class TeiObject(object):
    def __init__(self, filename):
        self.filename = filename
        #self._successfullyConverted = False
        self.soup = read_tei(filename)
        self._text = None
        self._title = ''
        self._abstract = ''
    

    def basename(self):
        base_name=os.path.basename(self.filename)
        if base_name.endswith('.tei'):
            return base_name[0:-4]
        elif base_name.endswith('tei.xml'):
            return base_name[0:-8]


    def authors(self):
        result = []
        if self.soup.analytic is None:
            return result
        
        authors_in_header = self.soup.analytic.find_all('author')

        for author in authors_in_header:
            persname = author.persName
            if not persname:
                continue
            # firstname = elem_to_text(persname.find("forename", type="first"))
            # middlename = elem_to_text(persname.find("forename", type="middle"))
            surname = elem_to_text(persname.surname)
            # person = Person(firstname, middlename, surname)
            if surname:
                result.append(surname)
        return result



    @property
    def title(self):
        if not self._title:
            if self.soup.find("title"):
                self._title = self.soup.title.getText()
        return self._title

    @property
    def abstract(self):
        if not self._abstract:
            if self.soup.find("abstract"):
                abstract = self.soup.abstract.getText(separator=' ', strip=True)
                self._abstract = abstract
        return self._abstract

    @property
    def text(self):
        if not self._text:
            divs_text = []
            if self.soup.find("body"):
                for div in self.soup.body.find_all("div"):
                    # div is neither an appendix nor references, just plain text.
                    if not div.get("type"):
                        div_text = div.get_text(separator=' ', strip=True)
                        divs_text.append(div_text)

                plain_text = " ".join(divs_text)
                self._text = plain_text
        return self._text
    
    #return two dict, both are of section: text format.
    # First one has only the text
    # second one has the references replaced with tokens(e.g., #b0, #b1)
    # Using the token we can track down particular reference from reference list
    def textWithSectionInfo(self):
        dictionary = {}
        seconddict={}

        if self.soup.find("body"):
            for div in self.soup.body.find_all("div"):
                newdiv=copy.copy(div)
                if not div.get("type"):
                    #setting dictionary key and value, one without ref, only text
                    dictionary[elem_to_text(div.head)]=elem_to_text(div)

                    #second dict with ref replaced with the ref token
                    for reftag in newdiv.find_all("ref"):
                        if(reftag.has_attr('type') and reftag.has_attr('target') and reftag['type']=="bibr"):
                            if re.search(r"^[0-9]*$",reftag.get_text().strip()):
                                # print(reftag.get_text())
                                newToken=int(reftag.get_text())
                                reftag.replace_with(" $b"+str(newToken)+" ")
                            else:
                                newToken=reftag['target']
                                reftag.replace_with(" "+str(newToken)+" ")

                         
                    textvalue=elem_to_text(newdiv)
                    textKey=elem_to_text(newdiv.head)
                    textvalue=textvalue.replace(textKey, "", 1)
                    seconddict[textKey]=textvalue
                    
        return dictionary, seconddict

    def footNote(self):
        footnote = {}
        if self.soup.find("body"):
            for note in self.soup.body.find_all("note"):
                if note.has_attr('n'):
                    footnote[int(note['n'])]=note.get_text()
        
        if footnote and len(footnote.keys())!=max(footnote.keys()):
            for i in range(1,max(footnote.keys())+1):
                if i not in footnote.keys():
                    if (i-1) in footnote.keys():
                        if footnote[i-1].find(str(i)) != -1:
                            splitted=footnote[i-1].split(str(i))
                            footnote[i]=splitted[-1]
                            footnote[i-1]=splitted[0]
                
            
        return footnote
    
    
    def reference(self):
        refer = {}
        if self.soup.find("back"):
            for div in self.soup.back.find_all("div"):
                if div.has_attr('type') and div['type']=="references": 
                    for ref in div.find_all("biblStruct"):
                        tempTitle = ""
                        tempLink = ""
                        tempTitleMono = ""
                        tempLinkMono = ""

                        if(ref.analytic is not None):
                            tempTitle=elem_to_text(ref.analytic.title)
                            if(ref.analytic.ptr is not None and ref.analytic.ptr.has_attr("target")):
                                tempLink=ref.analytic.ptr["target"]
                                
                        elif(ref.monogr is not None):
                            tempTitleMono=elem_to_text(ref.monogr.title)
                            if(ref.monogr.ptr is not None and ref.monogr.ptr.has_attr("target")):
                                tempLinkMono=ref.monogr.ptr["target"]
                                

                        #saving authors name
                        authorsList = []
                        if(ref.analytic is not None):
                            authors_in_header = ref.analytic.find_all('author')

                            for author in authors_in_header:
                                persname = author.persName
                                if not persname:
                                    continue
                                # firstname = elem_to_text(persname.find("forename", type="first"))
                                # middlename = elem_to_text(persname.find("forename", type="middle"))
                                surname = elem_to_text(persname.surname)
                                # person = Person(firstname, middlename, surname)
                                if surname:
                                    authorsList.append(surname)


                        #finally update the ref with ref tuple  
                        tempTuple = [tempTitle, tempLink, tempTitleMono, tempLinkMono]
                        tempTuple.extend(authorsList)
                        refer[ref['xml:id']]=tempTuple
                        
          
    
        return refer
    

    def table(self):
        table = {}
        if self.soup.find("body"):
            for figure in self.soup.body.find_all("figure"):
                if figure.has_attr('type') and figure['type']=="table":
                    table[figure['xml:id']]=elem_to_text(figure)

        return table
    
    
    def createJson(self):
        fulltext={}
        fulltext['title']=self.title
        fulltext['abstract']=self.abstract


        if not onlyAbstract:
            first, second=self.textWithSectionInfo()
            fulltext.update(second)
        
        footnote=self.footNote()
        reference=self.reference()
        table=self.table()
        
        report={}
        report['basename']=self.basename()
        report['fulltext']=len(fulltext.keys())
        report['footnote_size']=len(footnote.keys())
        if len(footnote.keys()) != 0:    
            report['footnote_max']=max(footnote.keys())
        report['reference']=len(reference.keys())
        report['authors'] = self.authors()
        
        dumper=[]
        dumper.append(report)
        dumper.append(fulltext)
        dumper.append(footnote)
        dumper.append(reference)
        dumper.append(table)
        
        posixvalue = Path.cwd().joinpath(self.basename())
        posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.json')

        with open(posixvalue, 'w') as output:
            json.dump(dumper, output, indent=4)



def process(teifile):
    basename=os.path.basename(teifile)
    if basename.endswith('tei.xml'):
        basename= basename[0:-8]
    
    posixvalue = Path.cwd().joinpath(basename)
    posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.json')

    if not Path(posixvalue).is_file() or overwrite:
        tei = TeiObject(teifile)
        tei.createJson()
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="path to the input directory(TEIXML)") 
    parser.add_argument("--output", default=None, help="path to the output directory(JSON)") 
    parser.add_argument("--numberOfCore", default=40, help="number of core") 
    parser.add_argument("--overwrite", action='store_true', help="Overwrite the output directory") 
    parser.add_argument("--onlyAbstract", action='store_true', help="Extract only abstract. Reference, Footnotes etc are extracted in all mode so JSON file format remain unchanged") 
    args = parser.parse_args()

    inpath = Path(args.input).resolve() 
    outpath = Path(args.output).resolve()
    overwrite = args.overwrite
    numberOfCore = int(args.numberOfCore)
    onlyAbstract = args.onlyAbstract

    print("TEI XML parser to process GROBID output XML files and generate JSON files for papers")
    print("with the following parameter")
    print("input directory:", inpath)
    print("output directory:", outpath)
    print("overwrite activated:", overwrite, "; Extraction mode, Only abstract:", onlyAbstract)
    print("number of CPU used:", numberOfCore)
    



    xmlfiles = sorted(Path.cwd().joinpath(inpath).glob('*.tei.xml'))
    os.chdir(Path.cwd().joinpath(outpath)) 
    pool = Pool(numberOfCore)
    pool.map(process, xmlfiles)

    print("--- %s seconds ---" % (time.time() - start_time))