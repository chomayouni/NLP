Report for Homework Zero - Normalization – Connor Homayouni
this report will also act as the readme file that is posted on my GitHub

introduction to the data selected:

To make the homework more interesting I decided to process and analyze the scripts from the OG Star Wars Trilogy. Specifically the scripts and not just a book that was explaining the theatrical version of the story. 
I chose a script because there are a bunch of script specific characters that would need to be taken into account when doing normalization and there is also the addition of extra text that wouldn't have been there and a book to supply more context to the reader. 
Overall it's a more diverse data set or at least that's the logic I used in my mind.


How to use the script(Methodology):
To use my program to analyze this scripts from the first Star Wars trilogy you simply have to download the  program homework_0.py and the text file “the_original_trilogy.txt” and make sure that they are added to the same directory. I've built the script so that as long as the text file you want to analyze is in the same directory as the program, you only have to type in the name .txt file and the program will fill in the rest of the path automatically.

There are several options that are available to use for analyzing the text:

"-l", "--lower", help="Convert all words to lowercase."
"-s", "--stem", help="Stem words."
"-sw", "--stop_word", help="Remove all stopwords from the final."
"-ph",'--search_phrase', help="Search for a specific phrase in the text."
"-ch", "--find_characters", help="Search for the total number of times the main characters are mentioned."

The first three options simply do what was required in the homework(lowercasing, stemming, and stop word removal) the final two options are customed to the Star Wars script although they could be adapted to fit other bodies of text. 
The search phrase option allows the user to type a string that they would like to search for within the text and it returns a graph that shows how many times that exact string appears. This option can also be modified with the first 
three options, though they will affect the amount of times a phrase appears. The last option was a fun one. I noticed that many of the most common words in the script word names of the characters. This makes, but I thought it might be
nice to be able to separate that data from the overall pool. The final option we'll return a graph that has the amount of times character names appear within the script and remove them from the overall pool of data. This list is specific to Star Wars episode 4, 
but you can easily go into the code and modify it to have a different list of names to exclude.
As long as the text you would like to analyze is in the same directory as the program you can run the command from the terminal with any combination of options. I have listed some examples below with their corresponding charts.

1.	python homework_0.py -l -s -sw -ph "the force" the_original_trilogy.txt  --find_characters
