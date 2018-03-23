package edu.usc.thevillagers.serversideagent;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.agent.AgentBrainExternal;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class CommandSpawn extends CommandBase {

	@Override
	public String getName() {
		return "s";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/s <script>";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 1)
			return;
		WorldServer world = server.worlds[0];
		BlockPos pos;
		if(args.length > 1)  
			pos = parseBlockPos(sender, args, 1, false);
		else
			pos = sender.getPosition();
		
		Agent agent = new Agent(world, args[0]);
    	agent.spawn(pos);
		agent.setBrain(new AgentBrainExternal("python python/"+args[0]+".py"));
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
}
