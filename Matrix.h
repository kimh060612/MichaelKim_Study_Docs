#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <iostream>
#include <cmath>
#include <memory.h>
#define det(a, b) (a+b)%2 == 0 ? 1 : -1
#define max(a, b) a > b ? a : b
#define EPS 1e-10

using namespace std;

class Matrix
{
private:
	Matrix expHelper(const Matrix&, int);
public:
	int row, col;
	double **MAT;

	Matrix();
	Matrix(int R, int C);
	Matrix(double **MAT_);
	Matrix(const Matrix& MAT_);
	~Matrix();

	inline double& operator()(int i, int j) { return this->MAT[i][j]; }

	Matrix& operator=(const Matrix&);
	Matrix& operator+=(const Matrix&);
	Matrix& operator-=(const Matrix&);
	Matrix& operator*=(const Matrix&);
	Matrix& operator*=(double num);
	Matrix& operator/=(double num);
	Matrix  operator^(int);
	Matrix& operator=(double **MAT_);

	double *PTR_ROW(int index);
	double *PTR_COL(int index);
	Matrix GetRow(int index);
	Matrix GetCol(int index);
	Matrix EigenValue(int Max_iter); // 1*N Matrix that each element is Eigen Value
	Matrix EigenVector(int Max_iter); // N*N Matrix that each Column is Eigen Vector
	void Identity();
	void show();
};

Matrix Trans(Matrix A);
Matrix operator+(const Matrix &op1, const Matrix &op2);
Matrix operator-(const Matrix &op1, const Matrix&op2);
Matrix operator*(const Matrix &op1, const Matrix&op2);
Matrix operator*(const Matrix&op1, int Constant);
Matrix operator*(double Constant, const Matrix&op1);
Matrix operator*(const Matrix& m, double num);
Matrix operator/(const Matrix& m, double num);
Matrix operator->*(const Matrix &op1, const Matrix&op2);

double Determinant(Matrix Target); // Same with Inverse_Matrix.
double CoFactor(Matrix Target, int p, int q);
Matrix Inverse(Matrix target); // Inverse Matrix Calc.
int argmax_1d(double *A, int s, int e);
Matrix RotPi(Matrix &M); // Rotate Filter 180 degree
Matrix solve(Matrix A, Matrix b);// Solve Ax = b linear Equation
void QR_Decompose(Matrix &A, Matrix &Q, Matrix &R); // QR-Decomposition function. Q, R will be N*N size, 0-initialize.

//-----------------------ConvNet 전용------------------------//
Matrix ZeroPadding(Matrix &M, int *padding);
Matrix Correlation(Matrix &op1, Matrix &op2, int *stride);  // op1: Image Map, op2: Filter
Matrix Convolution(Matrix &op1, Matrix &op2, int *stride);
Matrix MaxPooling(Matrix &op1, int *stride, int *PoolingSize); // Pooling Size는 무조건 크기가 2인 int형 배열

#endif // !__MATRIX_H__


