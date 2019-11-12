#set nth bit to 1; Count from zero
def SetBit( i , n ):
    return i | ( 1 << n )

#clear nth bit to 0
def ClearBit( i , n ):
    return i & ~ ( 1 << n )

#flip nth bit
def FlipBit( i , n ):
    return i ^ ( 1 << n )

#read nth bit
def ReadBit( i , n ):
    return ( i & ( 1 << n ) ) >> n

#count how many 1 bits
def PopCntBit( i ):
    return bin( i ).count('1')

#pick up n bits from kth bit
def PickBit( i , k , n ):
    return ( i & ( ( 2 ** n - 1 ) << k ) ) >> k

#circular bit shift left
def RotLBit( i , L , n ):
    return ( PickBit( i, 0, L-n ) << n ) + ( i >> ( L -n ) )

#circular bit shift right
def RotRBit( i , L , n ):
    return ( PickBit( i, 0, n ) << ( L - n ) ) + ( i >> n )