## extra file to test loading txt file into a list. function was also added in project1.py

def load_txt(filename):
    # opening the file in read mode
    my_file = open(filename, "r")

    # reading the file
    data = my_file.read()

    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    data_into_list = data.replace('\n', ' ').split()

    # make sure to close file
    my_file.close()

    return data_into_list

print( load_txt("stopwords.txt") )