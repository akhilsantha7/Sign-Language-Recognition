import re


def str2dict(strdict):
    """Convert key1=value1,key2=value2,... string into dictionary.
    :param strdict: key1=value1,key2=value2
    Note: This implementation overrides the original implementation
    in the neutronclient such that it is no longer required to append
    the key with a = to specify a corresponding empty value. For example,
    key1=value1,key2,key3=value3
    key1
    key1,key2
    will also be supported and converted to a dictionary with empty
    values for the relevant keys.
    """
    if not strdict:
        return {}
    return dict([kv.split('=', 1) if '=' in kv else [kv, ""]
                 for kv in strdict.split(',')])


def str2list(strlist):
    """Convert key1,key2,... string into list.
    :param strlist: key1,key2
    strlist can be comma or space separated.
    """
    if strlist is not None:
        strlist = strlist.strip(', ')
    if not strlist:
        return []
    return re.split("[, ]+", strlist)


def str2tuples(strlist, type=str):
    """Convert (key1, key2), (key1, key2),... string into tuple list.
    :param strlist: (key1, key2), (key1, key2),
    :param type: type to convert items to (default 'str').
    strlist can be comma or space separated.
    """
    if strlist is not None:
        strlist = strlist.strip(', ')
    if not strlist:
        return []
    
    tuples = re.findall(r'\([ ]?(\d*)[, ]+(\d*)[ ]?\)', strlist)
    tuples = [(type(x), type(y)) for (x, y) in tuples]

    if not (strlist.count('(') == strlist.count(')') == len(tuples)):
        raise ValueError("There was an error while reading tuples string")
    return tuples