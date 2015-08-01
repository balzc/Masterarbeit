package agents;

public class PlugAndForgetLoader {
	public double maxLoad;
	public PlugAndForgetLoader(double maxLoad){
		this.maxLoad = maxLoad;
	}
	public int actionQuery(double currentLoad){
		if(currentLoad < maxLoad){
			return 1;
		} else {
			return 0;
		}
	}
}
