package edu.usc.thevillagers.serversideagent.env.sensor;

import java.util.function.BooleanSupplier;

import edu.usc.thevillagers.serversideagent.agent.Actor;

/**
 * A one dimension sensor that encodes +1 or -1 depending on the provided predicate.
 */
public class SensorBoolean extends Sensor {
	
	private final BooleanSupplier predicate;

	public SensorBoolean(BooleanSupplier predicate) {
		super(1);
		this.predicate = predicate;
	}

	@Override
	public void sense(Actor actor) {
		values[0] = predicate.getAsBoolean() ? +1 : -1;
	}
}
