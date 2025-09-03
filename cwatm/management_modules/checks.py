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
def checkmap(name, value, map, flagmap, flagcompress, mapC):
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





    if flagmap:

        try:
            mapshape = input2str(map.shape[0]) + "x" + input2str(map.shape[1])
        except:
            mapshape = input2str(map.shape[0])

        if not(flagcompress):
            mapshape = input2str(map.shape[0]) + "x" + input2str(map.shape[1])
            numbernonmv = np.count_nonzero(~np.isnan(map))  # count nonmissing values
            numbermv = np.count_nonzero(np.isnan(map))  # count missing value (np.nan)
            #numbernan = "-"
            #numberzero = "-"
            numbernan = input2str(np.count_nonzero(np.isnan(map)))
            numberzero = input2str( map.shape[0] * map.shape[1] -  np.count_nonzero(map))
            numbernonzero = input2str(np.count_nonzero(map))

            compressF = "False"
            minmap = map[~np.isnan(map)].min()
            meanmap = map[~np.isnan(map)].mean()
            maxmap = map[~np.isnan(map)].max()

        else:
            numbernonmv = np.count_nonzero(~np.isnan(mapC))  # count nonmissing values
            numbermv = np.count_nonzero(np.isnan(mapC))  # count missing value (np.nan)

            compressF ="True"
            numbernan = input2str(np.count_nonzero(np.isnan(mapC)))
            numberzero = input2str(mapC.shape[0] - np.count_nonzero(mapC))
            numbernonzero = input2str(np.count_nonzero(mapC))

            minmap = mapC[~np.isnan(mapC)].min()
            meanmap = mapC[~np.isnan(mapC)].mean()
            maxmap = mapC[~np.isnan(mapC)].max()

        s.append(input2str(numbernonmv))
        s.append(input2str(numbermv))
        s.append(input2str(mapshape))
        s.append(compressF)
        s.append(numbernan)
        s.append(numberzero)
        s.append(numbernonzero)
        s.append(input2str(minmap))
        s.append(input2str(meanmap))
        s.append(input2str(maxmap))
        s.append(os.path.dirname(value))

    else:
        s.append("-")
        s.append("-")
        s.append("-")
        s.append("-")
        s.append("-")  # CompressF
        s.append("")
        s.append(input2str(float(map)))
        s.append("")

    if versioning['refvalue']:
        t = ["<30", "<80", "<20","<20","<10",">11", ">11", ">11", ">11", ">11", ">11", ">11", ">11", ">11", ">11", ">11", "<80"]
        h = ["Name", "File/Value", "Create Date","Ref Date","Same Date", "nonMV", "MV", "lon-lat", "Compress", "MV-comp", "Zero-comp", "NonZero", "min", "mean", "max",
             "Path"]
    else:
        t = ["<30","<80","<20"   ,">11",">11",">11",">11",">11",">11",">11",">11",">11", ">11",">11","<80"]
        h = ["Name","File/Value","Create Date","nonMV","MV", "lon-lat","Compress","MV-comp","Zero-comp","NonZero","min","mean","max","Path"]
    if checkmap.called == 1:
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

        for i in range(len(s)):
            s1 += f'{h[i]:{t[i]}}'
            if i<(len(s)-1):
                s1 += ","
            else:
                s1 += "\n"
        print(s1)
        versioning['check'] += s1

    s2 = ""
    for i in range(len(s)):
        s2 += f'{s[i]:{t[i]}}'
        if i < (len(s) - 1):
            s2 += ","
        else:
            s2 += "\n"
    print (s2)
    versioning['check'] += s2

    #print("%-30s%-40s%11i%11i%11i%11i%14.2f%14.2f%14.2f" %(s[0],s[1][-39:],s[2],s[3],s[8],s[7],s[4],s[5],s[6]))
    #print("%-30s%-40s%11s%11s%11s%11s%14s%14s%14s" % (s[0], s[1][-39:], s[2], s[3], s[8], s[7], s[4], s[5], s[6]))

    return

def save_check():

    save = False
    args = versioning['checkargs']
    if len(args)>1:
        if len(args) > 2 and args[1][-3:] == ".nc":
            if args[2][-4:] == ".csv":
                save = True
                savefile = args[2]
        else:
            if args[1][-4:] == ".csv":
                save = True
                savefile = args[3]
    if save:
        with open(savefile, 'w', encoding='utf-8') as f:
            f.write(versioning['check'])
    return








