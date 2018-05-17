package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import net.minecraft.util.math.MathHelper;
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
				Vec3d move = pos.subtract(prevPos).rotateYaw(yaw * (float) Math.PI / 180);
				values[0] = MathHelper.clamp((float) move.z / ct / .15F, -1, 1);
				values[1] = MathHelper.clamp((float) move.x / ct / .15F, -1, 1); //TODO more precise normalisation? not right derivative...
				return values;
			}
		};
	}
}
