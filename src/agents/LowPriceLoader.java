package agents;

public class LowPriceLoader {
	private double mq;
	private double vmin;
	private double qreq;
	private double qmax;
	public LowPriceLoader(double mq, double vmin, double qreq, double qmax){
		this.mq = mq;
		this.vmin = vmin;
		this.qreq = qreq;
		this.qmax = qmax;
	}
	public int actionQuery(double currentLoad, double price){
		if(currentLoad < qreq && price < vmin/qreq){
		return 1;
		}
		else if(currentLoad >= qreq && currentLoad < qmax && price < mq){
			return 1;
		}
		else{
			return 0;
		}
	}
}
