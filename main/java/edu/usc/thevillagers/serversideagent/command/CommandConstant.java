package edu.usc.thevillagers.serversideagent.command;

import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.command.WrongUsageException;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.text.TextComponentString;

/**
 * Command that alters some constant among: the number of ticks to skip
 * between agents' updates, agents' movement speed factor, agents' discrete
 * actions probability factor and agents' maximum pitch magnitude.
 */
public class CommandConstant extends CommandBase {
	
	public static int SKIP_TICK = 0;
	public static float AGENT_SPEED_FACTOR = 1;
	public static float ACTION_PROB_FACTOR = 1;
	public static float AGENT_PITCH_CLAMP = 90;
	
	@Override
	public String getName() {
		return "cst";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/cst <name> <value>";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 2) throw new WrongUsageException(getUsage(sender));
		switch(args[0]) {
		case "skip":
			SKIP_TICK = parseInt(args[1], 0);
			sender.sendMessage(new TextComponentString(String.format("skip=%d, period=%d, freq=%.1f", SKIP_TICK, SKIP_TICK+1, 20F / (SKIP_TICK+1))));
			break;
			
		case "speed":
			AGENT_SPEED_FACTOR = (float) parseDouble(args[1], 0);
			break;
			
		case "prob":
			ACTION_PROB_FACTOR = (float) parseDouble(args[1], 0);
			break;
			
		case "pitch":
			AGENT_PITCH_CLAMP = (float) parseDouble(args[1], 0);
			break;
		}
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
}
