package edu.usc.thevillagers.serversideagent.env.sensor;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorDummy;

/**
 * A sensor that gathers some communication vector from an agent.
 */
public class SensorComunication extends Sensor {
	
	private final ActuatorDummy com;

	public SensorComunication(ActuatorDummy com) {
		super(com.dim);
		this.com = com;
	}

	@Override
	public void sense(Agent agent) {
		for(int i = 0; i < dim; i++) values[i] = com.values[i];
	}
}
