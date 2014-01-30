// Copyright (c) 2014, Michael Boyle
// See LICENSE file for details

#ifndef ERRORS_HPP
#define ERRORS_HPP

// Note: These error codes are used in Quaternions.i.
//       If you change them here, change them there.

#define NotYetImplemented 0
// #define FailedSystemCall 1
// #define BadFileName 2
#define FailedGSLCall 3
// #define  4
// #define  5
// #define  6
// #define  7
// #define  8
// #define  9
#define ValueError 10
// #define BadSwitches 11
#define IndexOutOfBounds 12
// #define  13
// #define  14
#define VectorSizeMismatch 15
// #define MatrixSizeMismatch 16
// #define MatrixSizeAssumedToBeThree 17
#define NotEnoughPointsForDerivative 18
// #define EmptyIntersection 19
#define InfinitelyManySolutions 20
#define VectorSizeNotUnderstood 21
#define CannotExtrapolateQuaternions 22

#endif // ERRORS_HPP
