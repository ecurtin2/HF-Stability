DATE=`date +%Y-%m-%d:%H:%M:%S`
cat $@.cpp > $@_example.cpp
cat $@.log | awk 'BEGIN{printf("\n\n/* OUTPUT AS OF '$DATE':\n")} {printf(">>> %s\n", $0)} END{printf("*/")}' >> $@_example.cpp
rm $@.log
