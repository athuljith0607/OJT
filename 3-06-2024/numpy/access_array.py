import numpy as np
#created an array

array_2D =np.array(
    [
        [1,2,3],#0
        [4,5,6]#1
        ]
    )


#accessing  a single element
element =array_2D[1,2]
print("element at the index of [1,2]",element)

#access the second elemet in 1st array

element =array_2D[0,1]
print("element at the index of [0,1]",element)

#access by 1 row 
row =array_2D[0,:]
print ("first row: ",row)

#access second colums
column =array_2D[:,1]
print ("second column: ",column)

#slicing
# access the subarray with row of 0 and 1 ,column of 1 and 2
slice_array =array_2D[0:2,1:3]
print ("subarray : ",slice_array)


