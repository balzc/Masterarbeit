package mean;

import org.jblas.DoubleMatrix;

public class Zeromean extends Mean{
	public Zeromean(int dim){
		values = DoubleMatrix.zeros(dim);
	}
}
