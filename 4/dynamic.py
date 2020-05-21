import numpy as np
import pandas
import math



def main():

    df = pandas.read_csv('city1.csv', header = None)
    inputs = np.array(df.iloc[:,:].values.tolist())
   
    _nods = []
    _n = []
    _last_dest = 'F'
    _last_neighbor = [[_last_dest,0]]
    f_neighbor = []
    _path_nods = []
    
    for i in inputs:
        if(_n.count(i[0]) == 0):
            #if you arrive to f , add weight zero because theres no distance
            if i[0]== _last_dest:
                _nods.append([i[0],0])
                _n.append(i[0])
            else:
                _nods.append([i[0], math.inf]) 
                _n.append(i[0])
        if(_n.count(i[1]) == 0):
            if i[1]== _last_dest:
                _nods.append([i[1],0])
                _n.append(i[1])
            else:
                _nods.append([i[1], math.inf]) 
                _n.append(i[1])
# checked the last neighbers with f and add the weight of connection with f , فحص جيران ف كلهم وحط الوزن اللي بيناتهم وضافه بالاراي 
    for n in range(len(_n)-1):
        min_nods = []
        _new_neighbor = []
        for _last in _last_neighbor:
          

            min_nods = _last
            f_neighbor=[]

            for i in inputs:
                if i[0]==min_nods[0] :
                    f_neighbor.append(i[1:])
                    
                elif i[1]==min_nods[0] :
                    f_neighbor.append(i[::len(i)-1])
                    
            
            for _neig in f_neighbor:
              
                for x in _nods:
                    if x[0] == _neig[0]:
                        _nods_index = _nods.index(x)
                        break
                _w = min( (int(min_nods[1]) + int(_neig[1])), _nods[_nods_index][1])
                _nods[_nods_index][1] = _w
                _isExist = False
                for _new in _new_neighbor:
                    if (_new.count(_neig[0]) > 0):
                        _new[1] = _w
                        _isExist = True
                
                if not _isExist:
                    _new_neighbor.append([_neig[0], _w])

        _last_neighbor= _new_neighbor.copy()

    print(_nods)

    for _num_nods in _nods:
        _path_nods = []
        
        _last_neighbor = _num_nods
        _path_nods.append(_last_neighbor[0])
        while _last_neighbor[0] != _last_dest[0]:
            
            f_neighbor = []
            _check_neighbor=[]
            for i in inputs:
              
                if i[0] ==_last_neighbor[0] and _check_neighbor.count(i[1])==0:
                    f_neighbor.append(list(i[1:]))
                    _check_neighbor.append(i[1])
                        
                elif i[1] ==_last_neighbor[0] and _check_neighbor.count(i[0])==0:
                    f_neighbor.append(list(i[::len(i)-1]))
                    _check_neighbor.append(i[0])
            
            _isFirst = True
            for _neig in f_neighbor:
                _wight = 0
                for _w in _nods:
                    if _neig[0] == _w[0]:
                        _wight = _w[1]
                        break

                if _isFirst:
                    min_nods = _neig
                    min_nods[1] = str( int(_neig[1])+_wight)
                    _isFirst=False

                if int(min_nods[1]) > int(_neig[1])+_wight:
                    min_nods = _neig
                    min_nods[1] = str( int( _neig[1])+_wight)
            _last_neighbor = min_nods
            _path_nods.append(_last_neighbor[0])
        print(str(_path_nods) + ' : ' + str(_num_nods[1]))




if __name__ == '__main__':
     main()