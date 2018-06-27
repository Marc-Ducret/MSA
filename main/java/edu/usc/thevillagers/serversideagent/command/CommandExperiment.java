package edu.usc.thevillagers.serversideagent.command;

import java.util.List;

import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.env.EnvironmentManager;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.command.WrongUsageException;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import net.minecraftforge.fml.common.gameevent.TickEvent.Phase;

/**
 * Command that compiles a recording into a observation-action dataset
 */
public class CommandExperiment extends CommandBase {
	
	private EnvironmentManager envManager;
	
	public CommandExperiment(EnvironmentManager envManager) {
		this.envManager = envManager;
	}
	
	@Override
	public String getName() {
		return "exp";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/exp <env-type> <episodes> <x> <y> <z> <player-name>...";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 6) throw new WrongUsageException(getUsage(sender));
		String envType = args[0];
		int episodes = Integer.parseInt(args[1]);
		BlockPos pos = parseBlockPos(sender, args, 2, false);
		String envId = "exp";
		
		server.commandManager.executeCommand(sender, String.format("e add %s %s %d %d %d", envId, envType, pos.getX(), pos.getY(), pos.getZ()));
		String players = args[5];
		for(int i = 6; i < args.length; i++) players += ' ' + args[i];
		server.commandManager.executeCommand(sender, String.format("e player %s %s", envId, players));
		int size = 32;
		server.commandManager.executeCommand(sender, String.format("rec start %d %d %d %d %d %d", 
				pos.getX() - size, pos.getY() - 2, pos.getZ() - size,
				pos.getX() + size, pos.getY() + 8, pos.getZ() + size));
		Environment env = envManager.getEnv(envId);
		int[] ep = {0};
		envManager.addTickListener((phase) -> {
			if(env.time == 0 && phase == Phase.START) System.out.println(ep[0]+" done");
			if(phase == Phase.START && env.time == 0 && ep[0]++ >= episodes) {
				server.commandManager.executeCommand(sender, String.format("e remove %s", envId));
				server.commandManager.executeCommand(sender, "rec stop");
				return true;
			}
			return false;
		});
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
	
	@Override
	public List<String> getTabCompletions(MinecraftServer server, ICommandSender sender, String[] args,
			BlockPos targetPos) {
		if(args.length > 5) return getListOfStringsMatchingLastWord(args, server.getOnlinePlayerNames());
		return getListOfStringsMatchingLastWord(args);
	}
}
