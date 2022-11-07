import attribute_generator as ag

if __name__ == '__main__':
    filename = input('what is the name of the file?')
    soft = ag.AttrGen(filename)
    soft.run()
    print(soft.get_df14())
    
    
