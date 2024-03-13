datedbpediaids="2022.12.01"

# Dbpedia<->Wikidata
wget --no-check-certificate -O "ids.ttl.bz2" "https://downloads.dbpedia.org/repo/dbpedia/wikidata/sameas-all-wikis/${datedbpediaids}/sameas-all-wikis.ttl.bz2"
bzip2 -d ids.ttl.bz2
grep 'http://dbpedia.org/resource/' ids.ttl > ids_en.ttl
grep 'http://de.dbpedia.org/resource/' ids.ttl > ids_de.ttl

sed -i 's/<http:\/\/wikidata.dbpedia.org\/resource\///g ; s/> <http:\/\/www.w3.org\/2002\/07\/owl#sameAs> <http:\/\/dbpedia.org\/resource\// /g ; s/> .//g' ids_en.ttl
rm ids.ttl
mv ids_en.ttl data/wikidata_to_dbpedia_en.csv

# DBpedia redirects
wget --no-check-certificate -O "redirects_en.ttl.bz2" "https://downloads.dbpedia.org/repo/dbpedia/generic/redirects/${datedbpediaids}/redirects_lang%3den_transitive.ttl.bz2"
bzip2 -d redirects_en.ttl.bz2
sed -i 's/> <http:\/\/dbpedia.org\/ontology\/wikiPageRedirects> <http:\/\/dbpedia.org\/resource\// /g ; s/<http:\/\/dbpedia.org\/resource\///g ; s/> .//g' redirects_en.ttl
mv redirects_en.ttl data/redirects_en.csv

# Types
wget --no-check-certificate -O "types.ttl.bz2" "https://databus.dbpedia.org/dbpedia/mappings/instance-types/${datedbpediaids}/instance-types_lang=en_transitive.ttl.bz2"
bzip2 -d types.ttl.bz2
grep 'http://dbpedia.org/ontology' types.ttl > types_dbo_transitive.ttl


wget --no-check-certificate -O "types2.ttl.bz2" "https://databus.dbpedia.org/dbpedia/mappings/instance-types/${datedbpediaids}/instance-types_lang=en_specific.ttl.bz2"
bzip2 -d types2.ttl.bz2
grep 'http://dbpedia.org/ontology' types2.ttl > types_dbo_specific.ttl
cat types_dbo_transitive.ttl types_dbo_specific.ttl > types_dbo.ttl

grep '.wikidata.org/entity' types.ttl > types_wd.ttl
sed -i 's/> <http:\/\/www.w3.org\/1999\/02\/22-rdf-syntax-ns#type> <http:\/\/dbpedia.org\/ontology\// /g ; s/<http:\/\/dbpedia.org\/resource\///g ; s/> .//g' types_dbo.ttl
sed -i 's/> <http:\/\/www.w3.org\/1999\/02\/22-rdf-syntax-ns#type> <http:\/\/www.wikidata.org\/entity\// /g ; s/<http:\/\/dbpedia.org\/resource\///g ; s/> .//g' types_wd.ttl
rm types.ttl
mv types_dbo.ttl types_dbo.csv
mv types_wd.ttl types_wd.csv
rm ids_de.ttl