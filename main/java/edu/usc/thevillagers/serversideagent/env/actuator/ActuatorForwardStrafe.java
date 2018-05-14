package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import net.minecraft.util.math.Vec3d;

/**
 * Actuator of dimension 2 that enables horizontal movement.
 */
public class ActuatorForwardStrafe extends Actuator {

	public ActuatorForwardStrafe() {
		super(2);
	}

	@Override
	public void act(Actor actor) {
		actor.actionState.forward = values[0];
		actor.actionState.strafe = values[1];
	}

	@Override
	public Reverser reverser(Actor actor) {
		return new Reverser(actor) {
			
			private Vec3d prevPos;
			private float yaw;
			private int ct;
			
			@Override
			public void startStep() {
				ct = 0;
				prevPos = actor.entity.getPositionVector();
				yaw = actor.entity.rotationYaw;
			}
			
			@Override
			public void tick() {
				ct ++;
			}
			
			@Override
			public float[] endStep() {
				Vec3d pos = actor.entity.getPositionVector();
				Vec3d move = pos.subtract(prevPos).rotateYaw(-yaw);
				values[0] = (float) move.z / ct;
				values[1] = (float) move.x / ct;
				return values;
			}
		};
	}
}
