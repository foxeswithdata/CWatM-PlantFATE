# -------------------------------------------------------------------------
# Name:        checks if inputs are valid
# Purpose:
#
# Author:      burekpe
#
# Created:     16/05/2016
# Copyright:   (c) burekpe 2016
# -------------------------------------------------------------------------


from .globals import *
from netCDF4 import Dataset

def decompress(map):
    """
    Decompress 1D array without missing values to 2D array with missing values

    :param map: numpy 1D array as input
    :return: 2D array for displaying
    """

    dmap = maskinfo['maskall'].copy()
    dmap[~maskinfo['maskflat']] = map[:]
    dmap = dmap.reshape(maskinfo['shape'])

    # check if integer map (like outlets, lakes etc
    try:
        checkint = str(map.dtype)
    except:
        checkint = "x"
    if checkint == "int16" or checkint == "int32":
        dmap[dmap.mask] = -9999
    elif checkint == "int8":
        dmap[dmap < 0] = 0
    else:
        dmap[dmap.mask] = -9999

    return dmap

def counted(fn):
    """
    count number of times a subroutine is called

    :param fn:
    :return: number of times the subroutine is called
    """
    def wrapper(*args, **kwargs):
        wrapper.called += 1
        return fn(*args, **kwargs)
    wrapper.called = 0
    wrapper.__name__ = fn.__name__
    return wrapper


@counted
def checkmap(name, value, map):
    """
    check maps if the fit to the mask map

    :param name: name of the variable in settingsfile
    :param value: filename of the variable
    :param map: data (either a number or a 1D array)
    :param flagmap: indicates a 1D array or a number
    :param flagcompress: is there a compressed map available
    :param mapC: compressed map
    :return: -

    Todo:
        still to improve, this is work in progress!
    """

    def load_global_attribute(filename, attribute_name):
        if not os.path.exists(filename):
            return None

        try:
            with Dataset(filename, 'r') as nc_file:
                if attribute_name in nc_file.ncattrs():
                    return str(nc_file.getncattr(attribute_name))
                else:
                    return None
        except Exception:
            return None

    def input2str(inp):
        if isinstance(inp, str):
            return(inp)
        elif isinstance(inp, int):
            return f'{inp}'
        else:
            if inp < 100000:
                return f'{inp:.2f}'
            else:
                return f'{inp:.2E}'
    # ------------------------
    # if args[] is a netcdf then load this and analyse
    args = versioning['checkargs']
    if versioning['loadinput'] and len(args)>1:
        if args[1][-3:] == ".nc":
            # load discharge netcdf but only attribute version_inputfiles
            ver_input = load_global_attribute(args[1],"version_inputfiles")
            versioning['loadinput'] = False
            versioning['refvalue'] = True

            # put information on input data into dictorary
            versioning['checkinput'] = {}
            pairs = ver_input.split(';')
            for pair in pairs:
                if not pair.strip():
                    continue
                parts = pair.split(' ', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    date1 = parts[1].strip()
                else:
                    date1 = ""
                versioning['checkinput'][key] = date1





    # ----------------------------------
    # stored inputdate with date (addtoversiondate in data_handling.py)
    # (name, value, map):

    inputver =versioning['input'].split(";")
    # dictorary with each file and date
    inputv = {}
    for v in inputver[0:-1]:
        vv = v.split(" ")
        inputv[vv[0]] = vv[1] + " "+ vv[2]


    s = [name]
    #s.append(os.path.dirname(value))
    iv = os.path.basename(value)
    s.append(iv)
    # check for filename and get date
    createdate = inputv.get(iv, " ")
    s.append(createdate)

    # if a reference inputfile is used
    if versioning['refvalue']:
        refdate = versioning['checkinput'].get(iv, "")
        s.append(refdate)
        if refdate != "":
            if refdate == createdate:
                s.append("True")
            else:
                s.append("False")
        else:
            s.append(" ")

    # evaluate maps
    # if it is notr a number but a map (.tif, .nc, .map)
    flagmap = False
    if isinstance(map, np.ndarray):
        flagmap = True
        mapshape = map.shape
        # ifr compressed -> decompress
        if len(mapshape) < 2:
            map = decompress(map)
            mapshape = map.shape

    if flagmap:
        # if smaller than 0 or bigger than 1e20 => nan
        map = np.where(map<-100, np.nan, map)
        map = np.where(map > 1e20, np.nan, map)
        mapshape = input2str(map.shape[0]) + "x" + input2str(map.shape[1])

        #maskinfo['mask']
        # check if there are less valid cells than there should be compared to maskmap
        # reverse maskmap -> every valid cell has a True
        mask = ~maskinfo['mask']
        # count number of must cells
        numbermask = np.nansum(mask)
        vmap = ~np.isnan(map)
        andmap = mask & vmap
        # count number of cell in map
        numbermap = np.nansum(andmap)

        # if this is less the the must cell -> problem
        valid = "True"
        if numbermap < numbermask:
            valid = "False"


        numbernonzero = np.count_nonzero(map)
        numberzero = map.shape[0] * map.shape[1] - np.count_nonzero(map)

        minmap = map[~np.isnan(map)].min()
        meanmap = map[~np.isnan(map)].mean()
        maxmap = map[~np.isnan(map)].max()

        s.append(mapshape)
        s.append(input2str(int(numbermap)))
        s.append(valid)
        s.append(input2str(numberzero))
        s.append(input2str(numbernonzero))
        s.append("    ")
        s.append(input2str(minmap))
        s.append(input2str(meanmap))
        s.append(input2str(maxmap))
        s.append(os.path.dirname(value))

    # if it is a number
    else:
        #s.append(input2str(float(map)))
        for i in range(10):
            s.append("")




    # if it is checked against a discharge...nc
    if versioning['refvalue']:
        t = ["<30", "<80", "<20","<20","<10",">11", ">11", ">11", ">11", ">11", ">11", ">11", ">11", ">11", ">11", ">11", "<80"]
        h = ["Name", "File/Value", "Create Date","Ref Date","Same Date", "x-y", "number valid", "valid", "Zero values", "NonZero","-----",
             "min", "mean", "max", "Path"]
    # or without comparsion
    else:
        t = ["<30","<80","<20"   ,">11",">11",">11",">11",">11",">11",">11",">11",">11", ">11",">11","<80"]
        h = ["Name","File/Value","Create Date", "x-y", "number valid", "valid", "Zero values", "NonZero","-----",
             "min", "mean", "max", "Path"]

    # if checkmap is called for the first time
    if checkmap.called == 1:
        """
        s1= "----\n"
        s1 += "nonMV,non missing value in 2D map\n"
        s1 += "MV,missing value in 2D map\n"
        s1 += "lon-lat,longitude x latitude of 2D map\n"
        s1 += "CompressV,2D is compressed to 1D?\n"
        s1 += "MV-comp,missing value in 1D\n"
        s1 += "Zero-comp,Number of 0 in 1D\n"
        s1 += "NonZero,Number of non 0 in 1D\n"
        s1 += "min,minimum in 1D (or 2D)\n"
        s1 += "mean,mean in 1D (or 2D)\n"
        s1 += "max,maximum in 1D (or 2D)\n"
        s1 += "-----\n"
        """
        s1 =""
        # put all the header (keys) in a text line
        for i in range(len(s)):
            s1 += f'{h[i]:{t[i]}}'
            if i<(len(s)-1):
                s1 += ","
            else:
                s1 += "\n"
        print(s1)
        versioning['check'] += s1

    # put all the values in a text file
    s2 = ""
    for i in range(len(s)):
        s2 += f'{s[i]:{t[i]}}'
        if i < (len(s) - 1):
            s2 += ","
        else:
            s2 += "\n"
    versioning['check'] += s2
    s2 = str(checkmap.called) + " " + s2
    print (s2)

    return

def save_check():
    """
    Save the checked file
    """

    save = False
    checkmap.called = 0
    args = versioning['checkargs']
    if len(args)>1:
        if len(args) > 2 and args[1][-3:] == ".nc":
            if args[2][-4:] == ".csv":
                save = True
                savefile = args[2]
        else:
            if args[1][-4:] == ".csv":
                save = True
                savefile = args[1]
    if save:
        with open(savefile, 'w', encoding='utf-8') as f:
            f.write(versioning['check'])
    return








