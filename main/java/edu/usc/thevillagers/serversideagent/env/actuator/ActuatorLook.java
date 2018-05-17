package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.agent.EntityAgent;
import net.minecraft.util.math.MathHelper;

/**
 * Actuator of dimension 2 that enables horizontal movement.
 */
public class ActuatorLook extends Actuator {

	public ActuatorLook() {
		super(2);
	}

	@Override
	public void act(Actor actor) {
		actor.actionState.momentumYaw = values[0];
		actor.actionState.momentumPitch = values[1];
	}

	@Override
	public Reverser reverser(Actor actor) {
		return new Reverser(actor) {
			
			private float prevYaw, prevPitch;
			private int ct;
			
			@Override
			public void startStep() {
				ct = 0;
				prevYaw   = actor.entity.rotationYaw;
				prevPitch = actor.entity.rotationPitch;
			}
			
			@Override
			public void tick() {
				ct ++;
			}
			
			@Override
			public float[] endStep() {
				values[0] = MathHelper.clamp((actor.entity.rotationYaw   - prevYaw  ) / ct / EntityAgent.ROTATION_SPEED, -1, 1);
				values[1] = MathHelper.clamp((actor.entity.rotationPitch - prevPitch) / ct / EntityAgent.ROTATION_SPEED, -1, 1);
				return values;
			}
		};
	}
}
