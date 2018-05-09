package edu.usc.thevillagers.serversideagent.env.sensor;

import java.util.function.Function;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import net.minecraft.block.state.IBlockState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.BlockPos.MutableBlockPos;

/**
 * Sensor that encodes surrounding blocks in a box relative to the agent
 * defined by {@link #from} and {@link #to}.
 */
public class SensorBlocks extends Sensor {

	private final BlockPos from, to;
	private final Function<IBlockState, Float> encoding;
	
	public SensorBlocks(BlockPos from, BlockPos to, Function<IBlockState, Float> encoding) {
		super(	(to.getX() - from.getX() + 1) * 
				(to.getY() - from.getY() + 1) * 
				(to.getZ() - from.getZ() + 1));
		this.from = from;
		this.to = to;
		this.encoding = encoding;
	}

	@Override
	public void sense(Actor actor) {
		int offset = 0;
		BlockPos agentPos = actor.entity.getPosition();
		for(MutableBlockPos p : BlockPos.getAllInBoxMutable(from.add(agentPos), to.add(agentPos))) {
			values[offset++] = encoding.apply(actor.entity.world.getBlockState(p));
		}
	}
}
