package edu.usc.thevillagers.serversideagent.env.sensor;

import java.util.function.BooleanSupplier;

import edu.usc.thevillagers.serversideagent.agent.Agent;

public class SensorBoolean extends Sensor {
	
	private final BooleanSupplier predicate;

	public SensorBoolean(BooleanSupplier predicate) {
		super(1);
		this.predicate = predicate;
	}

	@Override
	public void sense(Agent agent) {
		values[0] = predicate.getAsBoolean() ? 1 : 0;
	}
}
