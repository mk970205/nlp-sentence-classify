def char_to_id(charlist):
    idlist = []
    for i in range(len(charlist)):
        char = charlist[i].lower()
        id = ord(char) - 96
        idlist.append(id)

    return idlist

print(type('a'))