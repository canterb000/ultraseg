import wfdb
import os

dblist = wfdb.getdblist()

for filename in dblist:
	
    print(filename[0])
    if len(filename) > 1:
        print("\t\t\t{}".format(filename[1]))



#wfdb.dldatabase('edb', "./edb")
#wfdb.dldatabase('ltstdb', "./ltstdb")
#wfdb.dldatabase('mitdb', "./mitdb")
#wfdb.dldatabase('nstdb', "./nstdb")
#wfdb.dldatabase('chfdb', "./chfdb")
#wfdb.dldatabase('ecgdmmld', "./ecgdmmld")
#wfdb.dldatabase('qtdb', "./qtdb")
wfdb.dldatabase('ltafdb', "./ltafdb")
wfdb.dldatabase('aftdb', "./aftdb")

#wfdb.dldatabase('qtdb', "./qtdb")
#wfdb.dldatabase('stdb', "./stdb")
