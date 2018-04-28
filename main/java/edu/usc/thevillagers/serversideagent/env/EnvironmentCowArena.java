package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.HighLevelAction;
import edu.usc.thevillagers.serversideagent.HighLevelAction.Phase;
import edu.usc.thevillagers.serversideagent.HighLevelAction.Type;
import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import net.minecraft.block.BlockColored;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityLiving;
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

	public final int size;

	protected BlockPos ref;

	private EntityCow[] cows;
	
	public EnvironmentCowArena() {
		this(5, 5);
	}

	public EnvironmentCowArena(int size, int nCows) {
		super(3 + nCows * 3, 2);
		this.size = size;
		cows = new EntityCow[nCows];
		allocator = new AllocatorEmptySpace(new BlockPos(-size, -1, -size), new BlockPos(size, 2, size));
	}
	
	@Override
	public void newAgent(Agent a) {
		super.newAgent(a);
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		ref = getOrigin();
	}

	@Override
	public void encodeObservation(Agent agent, float[] stateVector) {
		stateVector[0] = (float) (agent.entity.rotationYaw) / 360;
		stateVector[1] = (float) (agent.entity.posX - ref.getX()) / size;
		stateVector[2] = (float) (agent.entity.posZ - ref.getZ()) / size;
		for(int i = 0; i < cows.length; i++) {
			stateVector[3 + i * 3 + 0] = (float)(cows[i].posX - agent.entity.posX) / size;
			stateVector[3 + i * 3 + 1] = (float)(cows[i].posZ - agent.entity.posZ) / size;
			stateVector[3 + i * 3 + 2] = cows[i].getHealth() / .1F;
		}
	}

	@Override
	public void decodeAction(Agent agent, float[] actionVector) {
		agent.actionState.forward = actionVector[0];
		agent.actionState.strafe = actionVector[1];
//		agent.actionState.momentumYaw = actionVector[2];
	}

	@Override
	protected void stepAgent(Agent agent) throws Exception {
		agent.reward = 0;
		agent.actionState.action = null;
		for(int i = 0; i < cows.length; i++) {
			agent.reward -= cows[i].getHealth();
			if(cows[i].getHealth() > 0) {
				if(cows[i].getDistanceSq(agent.entity) < 1) {
					agent.actionState.action = new HighLevelAction(Type.HIT, Phase.INSTANT, agent.entity.getEntityId(), 
							EnumHand.MAIN_HAND, new ItemStack(Items.DIAMOND_SWORD), cows[i].getEntityId(), null, null, null);
					agent.reward += 5;
				}
			}
		}
		if(time >= 49) done = true;
	}
	
	@Override
	public void reset() {
		super.reset();
		for(Entity e : world.getEntitiesWithinAABB(Entity.class, new AxisAlignedBB(ref.add(-size, -1, -size), ref.add(size, +2, size))))
			if(!(e instanceof EntityLivingBase)) e.setDead();
		generate();
		for(int i = 0; i < cows.length; i++) {
			if(cows[i] != null)
				cows[i].setDead();
			cows[i] = new EntityCow(world);
			cows[i].setPosition(ref.getX() - size + world.rand.nextInt(2 * size - 3) + 2, ref.getY(), 
								ref.getZ() - size + world.rand.nextInt(2 * size - 3) + 2);
			cows[i].setHealth(.1F);
			cows[i].setDropItemsWhenDead(false);
			world.spawnEntity(cows[i]);
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
