package edu.usc.thevillagers.serversideagent.env;

import java.util.ArrayList;
import java.util.List;

import edu.usc.thevillagers.serversideagent.HighLevelAction;
import edu.usc.thevillagers.serversideagent.HighLevelAction.Phase;
import edu.usc.thevillagers.serversideagent.HighLevelAction.Type;
import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorForwardStrafe;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import net.minecraft.block.BlockColored;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityLivingBase;
import net.minecraft.entity.passive.EntityCow;
import net.minecraft.init.Blocks;
import net.minecraft.init.Items;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.item.ItemStack;
import net.minecraft.util.EnumHand;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentCowArena extends Environment {

	public int size;

	protected BlockPos ref;

	private List<EntityCow> cows;
	private int nCows;
	
	@Override
	public void readPars(float[] pars) {
		size = getRoundPar(pars, 0, 5);
		nCows = getRoundPar(pars, 1, 5);
		cows = new ArrayList<>();
		allocator = new AllocatorEmptySpace(new BlockPos(-size, -1, -size), new BlockPos(size, 2, size));
	}
	
	@Override
	protected void buildSensors() {
//		sensors.add(new SensorGaussian(1));
//		for(int i = 0; i < cows.length; i++) {
//			final int index = i;
//			sensors.add(new SensorPosition(size, 0, size, 
//					(a) -> cows[index].getPositionVector().subtract(a.entity.getPositionVector())));
//			sensors.add(new SensorBoolean(() -> cows[index].getHealth() > 0));
//		}
		entityDim = 3;
		entityMax = 100;
	}
	
	@Override
	protected void buildActuators() {
		actuators.add(new ActuatorForwardStrafe());
	}
	
	@Override
	public void encodeEntityObservation(Agent a, List<Float> obs) {
		super.encodeEntityObservation(a, obs);
		for(EntityCow cow : cows) {
			obs.add((float) ((cow.posX - a.entity.posX) / size));
			obs.add((float) ((cow.posZ - a.entity.posZ) / size));
			obs.add(cow.getHealth() > 0 ? +1F : -1F);
		}
	}
	
	@Override
	public void newActor(Actor a) {
		super.newActor(a);
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		ref = getOrigin();
	}

	@Override
	protected void stepActor(Actor actor) throws Exception {
		actor.reward = 0;
		actor.actionState.action = null;
		boolean oneAlive = false;
		actor.reward -= .1F;
		EntityCow closestAlive = null;
		for(EntityCow cow : cows) {
			if(cow.getHealth() > 0) {
				oneAlive = true;
				if(cow.getDistanceSq(actor.entity) < 2) {
					actor.actionState.action = new HighLevelAction(Type.HIT, Phase.INSTANT, actor.entity.getEntityId(), 
							EnumHand.MAIN_HAND, new ItemStack(Items.DIAMOND_SWORD), cow.getEntityId(), null, null, null);
					actor.reward += 5;
				}
				if(closestAlive == null || cow.getDistanceSq(actor.entity) < closestAlive.getDistanceSq(actor.entity)) closestAlive = cow;
			}
		}
		if(closestAlive != null) actor.reward -= closestAlive.getDistanceSq(actor.entity) / (size * size) * .1F;
		if(!oneAlive) {
			done = true;
			actor.reward = 20;
		}
		if(time >= 99) done = true;
	}
	
	@Override
	public void reset() {
		super.reset();
		for(Entity e : world.getEntitiesWithinAABB(Entity.class, new AxisAlignedBB(ref.add(-size, -1, -size), ref.add(size, +2, size))))
			if(!(e instanceof EntityLivingBase)) e.setDead();
		generate();
		for(EntityCow cow : cows) cow.setDead();
		cows.clear();
//		nCows = 2 + world.rand.nextInt(4);
		for(int i = 0; i < nCows; i++) {
			EntityCow cow = new EntityCow(world);
			cow.setPosition(ref.getX() - size + world.rand.nextInt(2 * size - 3) + 2, ref.getY(), 
								ref.getZ() - size + world.rand.nextInt(2 * size - 3) + 2);
			cow.setHealth(.1F);
			cow.setDropItemsWhenDead(false);
			world.spawnEntity(cow);
			cows.add(cow);
		}
	}
	
	private void generate() {
		for(int z =- size; z <= size; z++) {
			for(int x =- size; x <= size; x++) {
				world.setBlockState(ref.add(x, -1, z), Blocks.STAINED_GLASS.getDefaultState());
				if(Math.max(Math.abs(x), Math.abs(z)) == size)
						world.setBlockState(ref.add(x, 1, z), Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.BLACK));
			}
		}
	}
}
