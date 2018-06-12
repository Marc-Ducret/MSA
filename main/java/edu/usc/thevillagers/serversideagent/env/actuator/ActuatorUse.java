package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.HighLevelAction;
import edu.usc.thevillagers.serversideagent.ServerSideAgentMod;
import edu.usc.thevillagers.serversideagent.HighLevelAction.Type;
import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.recording.ActionListener;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.util.EnumHand;
import net.minecraft.util.math.RayTraceResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

public class ActuatorUse extends Actuator {
	
	private static final double REACH = 3;

	public ActuatorUse() {
		super(1);
	}

	@Override
	public void act(Actor actor) {
		if(values[0] < .5) {
			actor.actionState.action = null;
			return;
		}
		World world = actor.entity.world;
		Vec3d from = actor.entity.getPositionVector().addVector(0, actor.entity.getEyeHeight(), 0);
		Vec3d to = from.add(actor.entity.getLookVec().scale(REACH));
		RayTraceResult result = ServerSideAgentMod.rayTrace(world, from, to, true, actor.entity);
		if(result == null) return;
		switch(result.typeOfHit) {
		case BLOCK:
			actor.actionState.action = new HighLevelAction(
					HighLevelAction.Type.USE, 
					HighLevelAction.Phase.INSTANT, actor.entity.getEntityId(), 
					EnumHand.MAIN_HAND, actor.entity.getHeldItem(EnumHand.MAIN_HAND), 
					-1, result.getBlockPos(), result.sideHit, result.hitVec);
			break;
			
		case ENTITY: // TODO test
			actor.actionState.action = new HighLevelAction(
					HighLevelAction.Type.USE, 
					HighLevelAction.Phase.INSTANT, actor.entity.getEntityId(), 
					EnumHand.MAIN_HAND, actor.entity.getHeldItem(EnumHand.MAIN_HAND), 
					result.entityHit.getEntityId(), null, null, result.hitVec);
			break;
			
		default:
			actor.actionState.action = null;
			break;
		}
	}

	@Override
	public Reverser reverser(Actor actor, WorldRecordReplayer replay) {
		return new Reverser(actor) {
			
			private ActionListener listener = (action) -> {
				if(action.actionType == Type.USE) values[0] = 1;
			};
			
			@Override
			public void startStep() {
				values[0] = 0;
				replay.actionListeners.add(listener);
			}
			
			@Override
			public void tick() {
			}
			
			@Override
			public float[] endStep() {
				replay.actionListeners.remove(listener);
				return values;
			}
		};
	}
}
