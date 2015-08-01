package agents;

import org.jblas.DoubleMatrix;

public class SortedMinLoader {
	public int[] policy;
	public void setup(DoubleMatrix predictedPrices, double qmax,double qinitial){
		policy = new int[predictedPrices.rows];
		double[] currentMins = new double[(int)(qmax-qinitial+1)];
		int[] indexes = new int[(int)(qmax-qinitial+1)];
		for(int j = 0; j < indexes.length; j++){
			double currentMin = Double.POSITIVE_INFINITY;

			for(int i = 0; i < predictedPrices.rows; i++){
				if(j > 0){
					if(predictedPrices.get(i,0) < currentMin && predictedPrices.get(i,0) >= currentMins[j-1] && indexes[j-1] != i ){
						currentMin = predictedPrices.get(i,0);
						indexes[j] = i;
						currentMins[j] = predictedPrices.get(i,0);
					}
				}
				else{
					if(predictedPrices.get(i,0) < currentMin){
						currentMin = predictedPrices.get(i,0);
						indexes[j] = i;
						currentMins[j] = predictedPrices.get(i,0);
					}
				}
			}
		}
		for(int j = 0; j < indexes.length; j++){
			policy[indexes[j]] = 1;
		}
	}
}
