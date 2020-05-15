#include "Matrix.h"

Matrix::Matrix()
{
}

Matrix::Matrix(int R, int C)
{
	this->row = R;
	this->col = C;

	this->MAT = new double*[R];
	for (int i = 0; i < R; i++)
	{
		this->MAT[i] = new double[C]();
	}
}

Matrix::Matrix(double **MAT_)
{
	if (_msize(MAT_) / sizeof(double) == 0)
	{
		this->MAT = MAT_;
	}
	else
	{
		this->row = _msize(MAT_) / sizeof(double*);
		this->col = _msize(MAT_[0]) / sizeof(double);
		this->MAT = new double*[this->row];
		for (int i = 0; i < this->row; i++)
		{
			this->MAT[i] = new double[this->col]();
			for (int j = 0; j < this->col; j++)
			{
				this->MAT[i][j] = MAT_[i][j];
			}
		}
	}

}

Matrix::Matrix(const Matrix & MAT_)
{
	this->row = MAT_.row;
	this->col = MAT_.col;
	this->MAT = new double*[MAT_.row];
	for (int i = 0; i < this->row; i++)
	{
		this->MAT[i] = new double[MAT_.col];
	}

	for (int i = 0; i < this->row; i++)
	{
		for (int j = 0; j < this->col; j++)
		{
			this->MAT[i][j] = MAT_.MAT[i][j];
		}
	}

}

Matrix::~Matrix()
{
	if (this->MAT != NULL)
	{
		for (int i = 0; i < this->row; i++)
		{
			delete[] this->MAT[i];
		}
		delete[] this->MAT;

		this->row = 0;
		this->col = 0;
	}
}

Matrix &Matrix::operator=(double **MAT_)
{
	this->row = _msize(MAT_) / sizeof(double*);
	this->col = _msize(MAT_[0]) / sizeof(double);
	this->MAT = MAT_;
	return *this;
}


Matrix operator->*(const Matrix &op1, const Matrix &op2)
{
	if (op2.row != op1.row || op2.col != op1.col)
	{
		cout << "Error: Shape Fault!" << endl;
	}
	else
	{
		Matrix res(op2.row, op2.col);

		for (int i = 0; i < op1.row; i++)
		{
			for (int j = 0; j < op1.col; j++)
			{
				res.MAT[i][j] = op2.MAT[i][j] * op1.MAT[i][j];
			}
		}
		return res;
	}
}

Matrix Trans(Matrix A)
{
	Matrix res(A.col, A.row);

	for (int i = 0; i < A.col; i++)
	{
		for (int j = 0; j < A.row; j++)
		{
			res.MAT[i][j] = A.MAT[j][i];
		}
	}

	return res;
}

double * Matrix::PTR_ROW(int index)
{
	if (index <0 || index > this->row)
	{
		cout << "index is not in the range" << endl;
		return NULL;
	}
	else
	{
		double *A = new double[this->row];
		for (int i = 0; i < this->col; i++)
		{
			A[i] = this->MAT[index][i];
		}
		return A;
	}
}

double * Matrix::PTR_COL(int index)
{
	if (index <0 || index > this->col)
	{
		cout << "index is not in the range" << endl;
		return NULL;
	}
	else
	{
		double *A = new double[this->row];
		for (int i = 0; i < this->row; i++)
		{
			A[i] = this->MAT[i][index];
		}
		return A;
	}
}

Matrix Matrix::GetRow(int index)
{
	if (index <0 || index > this->row)
	{
		throw domain_error("Error: Index out of range!");
	}
	Matrix res(1, this->col);
	for (int i = 0; i < this->col; i++)
	{
		res(0, i) = this->MAT[index][i];
	}
	return res;
}

Matrix Matrix::GetCol(int index)
{
	if (index <0 || index > this->col)
	{
		throw domain_error("Error: Index out of range!");
	}
	Matrix res(this->row, 1);
	for (int i = 0; i < this->row; i++)
	{
		res(i, 0) = this->MAT[i][index];
	}
	return res;
}

Matrix Matrix::EigenValue(int Max_iter)
{
	if (this->row != this->col) throw domain_error("Error: Invaild matrix size");
	int N = this->row;
	Matrix EigenVal(1,N);
	Matrix A(N, N);
	A = *this;

	for (int epoch = 0; epoch < Max_iter; epoch)
	{
		Matrix Q(N, N);
		Matrix R(N, N);
		QR_Decompose(A, Q, R);
		A = R * Q;
	}
	for (int i = 0; i < N; i++)
	{
		EigenVal(0, i) = A(i, i);
	}

	return EigenVal;
}

Matrix Matrix::EigenVector(int Max_iter)
{
	if (this->row != this->col) throw domain_error("Error: Invaild matrix size");
	int N = this->row;
	Matrix EigenVal = this->EigenValue(Max_iter);
	Matrix EigenVec(N,N);

	for (int i = 0; i < N; i++)
	{
		Matrix I(N, N);
		I.Identity();
		Matrix Tmp = (*this) - EigenVal(0,i)*I;
		Matrix Zeros(N, 1);
		Matrix EigenROW = solve(*this, Zeros);
		for (int j = 0; j < N; j++)
		{
			EigenVec(j, i) = EigenROW(j,0);
		}
	}

	return EigenVec;
}

void Matrix::show()
{
	for (int i = 0; i < this->row; i++)
	{
		for (int j = 0; j < this->col; j++)
		{
			cout << this->MAT[i][j] << "  ";
		}
		cout << endl;
	}
}


void Matrix::Identity()
{
	if (this->col != this->row)
	{
		cout << "Cannot make Identity" << endl;
	}
	else
	{
		int n = this->row;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (i == j)
				{
					this->MAT[i][j] = 1;
				}
				else
				{
					this->MAT[i][j] = 0;
				}

			}
		}
	}

}

Matrix& Matrix::operator+=(const Matrix& m)
{
	for (int i = 0; i < this->row; ++i)
	{
		for (int j = 0; j < this->col; ++j)
		{
			this->MAT[i][j] += m.MAT[i][j];
		}
	}
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& m)
{
	for (int i = 0; i < this->row; ++i)
	{
		for (int j = 0; j < this->col; ++j)
		{
			this->MAT[i][j] -= m.MAT[i][j];
		}
	}
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& m)
{
	Matrix temp(this->row, m.col);
	for (int i = 0; i < temp.row; ++i)
	{
		for (int j = 0; j < temp.col; ++j)
		{
			for (int k = 0; k < this->col; ++k)
			{
				temp.MAT[i][j] += (this->MAT[i][k] * m.MAT[k][j]);
			}
		}
	}
	return (*this = temp);
}

Matrix& Matrix::operator*=(double num)
{
	for (int i = 0; i < this->row; ++i)
	{
		for (int j = 0; j < this->col; ++j)
		{
			this->MAT[i][j] *= num;
		}
	}
	return *this;
}


Matrix& Matrix::operator/=(double num)
{
	for (int i = 0; i < this->row; ++i)
	{
		for (int j = 0; j < this->col; ++j)
		{
			this->MAT[i][j] /= num;
		}
	}
	return *this;
}

Matrix Matrix::operator^(int num)
{
	Matrix temp(*this);
	return expHelper(temp, num);
}

Matrix Matrix::expHelper(const Matrix& m, int num)
{
	if (num == 0)
	{
		Matrix A(m.row, m.row);
		A.Identity();
		return A;
	}
	else if (num == 1)
	{
		return m;
	}
	else if (num % 2 == 0)
	{  // num is even
		return expHelper(m * m, num / 2);
	}
	else
	{                    // num is odd
		return m * expHelper(m * m, (num - 1) / 2);
	}
}

Matrix& Matrix::operator=(const Matrix& m)
{
	if (this == &m)
	{
		return *this;
	}

	if (this->row != m.row || this->col != m.col)
	{
		if (this->row > 0 && this->col > 0)
		{
			for (int i = 0; i < this->row; ++i)
			{
				delete[] this->MAT[i];
			}
			delete[] this->MAT;
		}

		this->row = m.row;
		this->col = m.col;

		this->MAT = new double*[this->row];
		for (int i = 0; i < this->row; i++)
		{
			this->MAT[i] = new double[this->col];
		}
	}

	for (int i = 0; i < this->row; ++i)
	{
		for (int j = 0; j < this->col; ++j)
		{
			this->MAT[i][j] = m.MAT[i][j];
		}
	}
	return *this;
}

double Determinant(Matrix Target) // Same with Inverse_Matrix.
{
	try
	{
		int n;
		double Det = 0;
		if (Target.row != Target.col) { int e = 1; throw(e); }
		n = Target.col;
		Matrix SUBMAT(n - 1, n - 1);
		if (n == 2) return (Target.MAT[0][0] * Target.MAT[1][1] - Target.MAT[0][1] * Target.MAT[1][0]);
		else
		{
			for (int c = 0; c < n; c++)
			{
				int subi = 0;
				for (int i = 1; i < n; i++)
				{
					int subj = 0;
					for (int j = 0; j < n; j++)
					{
						if (j == c)continue;
						SUBMAT.MAT[subi][subj] = Target.MAT[i][j];
						subj++;
					}
					subi++;
				}
				Det = Det + (pow(-1, c) * Target.MAT[0][c] * Determinant(SUBMAT));
			}
		}
		return Det;
	}
	catch (int e)
	{
		cout << "Cannot get the Determinant. this is not the squre matrix" << endl;
	}
}

double CoFactor(Matrix Target, int p, int q)
{
	try
	{
		int n;
		if (Target.col != Target.row) { int e = 1; throw(e); }
		n = Target.col;
		Matrix SUB(n - 1, n - 1);
		int suba = 0;
		for (int i = 0; i < n; i++)
		{
			if (i == p)continue;
			int subb = 0;
			for (int j = 0; j < n; j++)
			{
				if (j == q)continue;
				else
				{
					SUB.MAT[suba][subb] = Target.MAT[i][j];
					subb++;
				}
			}
			suba++;
		}
		return pow(-1, p + q)*Determinant(SUB);
	}
	catch (int e)
	{
		cout << "Cannot get the cofactor. this is not the squre matrix" << endl;
	}
}

Matrix Inverse(Matrix target)
{
	try
	{
		double DET = Determinant(target);
		if (target.col != target.row || DET == double(0))
		{
			int e = 1;
			throw (e);
		}
		else
		{
			int n = target.col;
			Matrix RES(n, n);
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					RES.MAT[j][i] = CoFactor(target, i, j);
				}
			}
			RES /= DET;
			return RES;
		}
	}
	catch (int e)
	{
		cout << "Cannot get the Inverse Matrix." << endl;
	}
}

int argmax_1d(double *A, int s, int e)
{
	int max = -987654321;
	for (int i = s; i < e; i++)
	{
		if (max < A[i])max = i;
	}
	return max;
}

Matrix RotPi(Matrix & M)
{
	Matrix res(M.row, M.col);

	for (int i = 0; i < M.row; i++)
	{
		for (int j = 0; j < M.col; j++)
		{
			res(M.row - 1 - i, M.col - 1 - j) = M(i, j);
		}
	}

	return res;
}

Matrix solve(Matrix A, Matrix b)
{
	// Gaussian elimination
	for (int i = 0; i < A.row; ++i) {
		if (A(i,i) == 0) {
			// pivot 0 - throw error
			throw domain_error("Error: the coefficient matrix has 0 as a pivot. Please fix the input and try again.");
		}
		for (int j = i + 1; j < A.row; ++j) {
			for (int k = i + 1; k < A.col; ++k) {
				A(j,k) -= A(i,k) * (A(j,i) / A(i,i));
				if (A(j,k) < EPS && A(j,k) > -1 * EPS)
					A(j,k) = 0;
			}
			b(j,0) -= b(i,0) * (A(j,i) / A(i,i));
			if (A(j,0) < EPS && A(j,0) > -1 * EPS)
				A(j,0) = 0;
			A(j,i) = 0;
		}
	}

	// Back substitution
	Matrix x(b.row, 1);
	x(x.row - 1,0) = b(x.row - 1,0) / A(x.row - 1,x.row - 1);
	if (x(x.row - 1,0) < EPS && x(x.row - 1,0) > -1 * EPS)
		x(x.row - 1,0) = 0;
	for (int i = x.row - 2; i >= 0; --i) {
		int sum = 0;
		for (int j = i + 1; j < x.row; ++j) {
			sum += A(i,j) * x(j,0);
		}
		x(i,0) = (b(i,0) - sum) / A(i,i);
		if (x(i,0) < EPS && x(i, 0) > -1 * EPS)
			x(i, 0) = 0;
	}

	return x;
}

void QR_Decompose(Matrix & A, Matrix & Q, Matrix & R)
{
	if (A.col == A.row) throw domain_error("Error: The matrix size doesn't fit!");

	int n = A.row;
	for (int i = 0; i < n; i++)
	{
		Matrix A_i = A.GetCol(i); // 1*N
		Matrix Q_i = A_i;
		for (int j = 0; j < i; j++)
		{
			Matrix Q_j = Q.GetRow(j); // N*1
			R(j, i) = (A_i * Q_j)(0,0);
			Matrix tmp = R(j,i)*Q_j;
			Q_i = Q_i - tmp;
		}
		double sum = 0;
		for (int j = 0; j < n; j++)
		{
			sum += Q_i(j, 0)*Q_i(j, 0);
		}
		double Ravg = sqrt(sum);
		R(i, i) = Ravg;
		Q_i /= R(i, i);
		for (int j = 0; j < n; j++)
		{
			Q(i, j) = Q_i(j, 0);
		}
	}

	Q = Trans(Q);
}

Matrix ZeroPadding(Matrix & M, int *padding) // 오늘 내일?
{
	Matrix RES(M.row + 2*padding[0], M.col + 2*padding[1]);
	int OutH = M.row + 2 * padding[0];
	int OutW = M.col + 2 * padding[1];

	for (int i = 0, ii = 0; i < OutH; i++)
	{
		for (int j = 0, jj = 0; j < OutW; j++)
		{
			if ((i < padding[0] || i >= M.row) || (j < padding[1] || j >= M.col)) RES(i, j) = 0;
			else
			{
				RES(i, j) = M(ii, jj);
				ii++;
				jj++;
			}
		}
	}

	return RES;
}

Matrix Correlation(Matrix & op1, Matrix & op2, int *stride) // 이미 패딩된 값이 들어온다.
{
	int Out_H, Out_W;
	Out_H = ((op1.row - op2.row) / stride[0]) + 1;
	Out_W = ((op1.col - op2.col) / stride[1]) + 1;

	Matrix res(Out_H, Out_W);

	for (int i = 0; i < Out_H; i++)
	{
		for (int j = 0; j < Out_W; j++)
		{
			double sum = 0.;
			for (int p = 0; p < op2.row; p++)
			{
				for (int q = 0; q < op2.col; q++)
				{
					sum += op1(i + stride[0] * p, j + stride[1] * q) * op2(p, q);
				}
			}
			res(i, j) = sum;
		}
	}

	return res;
}

Matrix Convolution(Matrix & op1, Matrix & op2, int *stride)
{
	Matrix tmp = RotPi(op2);
	Matrix res = Correlation(op1, tmp, stride);
	return res;
}

Matrix MaxPooling(Matrix & op1, int *stride, int * PoolingSize) // 오늘 내일? 이것도 오늘 내일? 잘 모르겠다 일단 Q-learning을 다 짜고 생각을 하자.
{
	int PoolingH = PoolingSize[0];
	int PoolingW = PoolingSize[1];
	int OutH = ((op1.row - PoolingH) / stride[0]) + 1;
	int OutW = ((op1.col - PoolingW) / stride[1]) + 1;
	Matrix res(OutH, OutW);

	for (int i = 0; i < OutH; i++)
	{
		for (int j = 0; j < OutW; j++)
		{
			double Max = -987654321.;
			for (int p = 0; p < PoolingH; p++)
			{
				for (int q = 0; q < PoolingW; q++)
				{
					Max = max(Max, op1(i + stride[0] * p, j + stride[1] * q));
				}
			}
			res(i, j) = Max;
		}
	}

	return res;
}

Matrix operator+(const Matrix& m1, const Matrix& m2)
{
	Matrix temp(m1);
	return (temp += m2);
}

Matrix operator-(const Matrix& m1, const Matrix& m2)
{
	Matrix temp(m1);
	return (temp -= m2);
}

Matrix operator*(const Matrix& m1, const Matrix& m2)
{
	Matrix temp(m1);
	return (temp *= m2);
}

Matrix operator*(const Matrix & op1, int Constant)
{
	Matrix tmp(op1);
	return (tmp *= Constant);
}

Matrix operator*(const Matrix& m, double num)
{
	Matrix temp(m);
	return (temp *= num);
}

Matrix operator*(double Constant, const Matrix& m)
{
	return (m * Constant);
}

Matrix operator/(const Matrix& m, double num)
{
	Matrix temp(m);
	return (temp /= num);
}