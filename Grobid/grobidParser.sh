export PDFDir="../data/PDFs/"
export TEIXMLDir="../data/TEIXMLs/"
export JSONDir="../data/JSONs/"

# running Grobid multiple times to make sure none of the PDF was skipped
# it will not overwrite
# for i in {1..3}
# do
#     # n is the number of concurrent worker
#     python grobid-client-python/grobid-client.py --input $PDFDir --output $TEIXMLDir --n 80 processFulltextDocument --config grobid-client-python/config.json
# done

# TEIXML to JSON parser
python TEIXMLparser.py --input $TEIXMLDir --output $JSONDir --numberOfCore 40


