Documentation
=============

in "user.txt":

put in put in values within the range

DATASET STRUCTURE:

To ensure that all the right endpoints are run correctly call the endpoints like this
	
	for the download endpoint:
		replace the "<output>" tag with "total_game_data2.csv"
	
	for the data_separation endpoint:
		replace the "<datafile>" tag with "total_game_data2.csv"

	for the normalize endpoint:
		replace the "<datafile>" tag with "altered_total.csv"
	
	for the shuffle_plot endpoint:
		replace the "<altered_csv>" endpoint with  "altered_total.csv"
		and the "<labels>" with "class.csv"

	if everything is run like this then it should work

