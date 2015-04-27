package mdp;

public class MDP {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int states = 4;
		int timesteps = 10;
		double[][][] actions = {{{0,1,0,0},{0,0,1,0}},{{0.5,0,0,0.5}},{{0.2,0,0,0.8}},{{0,0,0,1}}};//new float[states][states][states];
		double[] rewards = new double[states];
		rewards[0] = 0;
		rewards[1] = 1;
		rewards[2] = 0;
		rewards[3] = 2;

		solveMDP(states, timesteps, actions, rewards);
	}

	
	public static int[][] solveMDP(int states, int timesteps, double[][][] actions,double[] rewards)
	{
		double[][] expectedRewards = new double[states][timesteps];
		int[][] policy = new int[states][timesteps];
		for(int s = 0; s < states; s++){
			expectedRewards[s][timesteps-1] = rewards[s];
		}
		for(int t = timesteps-2; t > -1; t--){
			for(int s = 0; s < states; s++){
				double maxReward = -9999990;
				int bestAction = 0;
				for(int o = 0; o < actions[s].length; o++){
					double eReward = rewards[s];
					for(int i = 0; i < actions[s][o].length; i++){
						eReward += actions[s][o][i] * expectedRewards[i][t+1];
						System.out.println(eReward);

					}
					if(maxReward < eReward){
						maxReward = eReward;
						bestAction = o;
						System.out.println("updated: " + maxReward);
					}
					System.out.println("action: " + o + " done");

				}
				expectedRewards[s][t] = maxReward;
				policy[s][t] = bestAction;
				System.out.println("state: " + s + " done");

			}
			System.out.println("timestep: " + t + " done");

		}
		for(int i = 0; i < states; i++){
			for(int o = 0; o < timesteps; o++){
				System.out.print(policy[i][o] + " ");
			}
			System.out.println();
		}
		for(int i = 0; i < states; i++){
			for(int o = 0; o < timesteps; o++){
				System.out.print(expectedRewards[i][o] + " ");
			}
			System.out.println();
		}
		return policy;
	}
}
