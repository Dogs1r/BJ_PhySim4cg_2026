#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <glm/glm.hpp>
#include <iostream>
#include <utility>
#include <vector>
#include "Labs/2-FluidSimulation/FluidSimulator.h"
#include "spdlog/spdlog.h"

namespace VCX::Labs::Fluid {
	namespace {
		constexpr float kEps = 1e-6f;

		inline float clamp01(float v) {
			return std::max(0.0f, std::min(1.0f, v));
		}
	} // namespace

	inline int Simulator::index2GridOffset(glm::ivec3 index) {
		return index.x * (m_iCellY * m_iCellZ) + index.y * m_iCellZ + index.z;
	}

	inline bool Simulator::isValidVelocity(int i, int j, int k, int dir) {
		if (i < 0 || j < 0 || k < 0 || i >= m_iCellX || j >= m_iCellY || k >= m_iCellZ)
			return false;

		const int idx = index2GridOffset(glm::ivec3(i, j, k));
		if (m_s[idx] == 0.0f)
			return false;

		if (dir == 0 && i + 1 < m_iCellX) {
			const int idxNext = index2GridOffset(glm::ivec3(i + 1, j, k));
			return m_s[idxNext] > 0.0f;
		}
		if (dir == 1 && j + 1 < m_iCellY) {
			const int idxNext = index2GridOffset(glm::ivec3(i, j + 1, k));
			return m_s[idxNext] > 0.0f;
		}
		if (dir == 2 && k + 1 < m_iCellZ) {
			const int idxNext = index2GridOffset(glm::ivec3(i, j, k + 1));
			return m_s[idxNext] > 0.0f;
		}
		return false;
	}

	void Simulator::integrateParticles(float timeStep) {
		for (int p = 0; p < m_iNumSpheres; ++p) {
			m_particleVel[p] += gravity * timeStep;
			m_particlePos[p] += m_particleVel[p] * timeStep;
		}
	}

	void Simulator::pushParticlesApart(int numIters, float timeStep) {
		if (m_iNumSpheres <= 1)
			return;

		std::vector<int> sortedParticleIds(m_iNumSpheres, 0);
		std::vector<glm::vec3> deltaVel(m_iNumSpheres, glm::vec3(0.0f));
		const float invDt = 1.0f / std::max(timeStep, 1e-6f);

		auto buildSpatialHash = [&]() {
			std::fill(m_hashtableindex.begin(), m_hashtableindex.end(), 0);

			for (int p = 0; p < m_iNumSpheres; ++p) {
				glm::vec3 g = (m_particlePos[p] + glm::vec3(0.5f)) * m_fInvSpacing;
				int i       = std::clamp(static_cast<int>(std::floor(g.x)), 0, m_iCellX - 1);
				int j       = std::clamp(static_cast<int>(std::floor(g.y)), 0, m_iCellY - 1);
				int k       = std::clamp(static_cast<int>(std::floor(g.z)), 0, m_iCellZ - 1);
				int cell    = index2GridOffset(glm::ivec3(i, j, k));
				m_hashtable[p] = cell;
				m_hashtableindex[cell + 1] += 1;
			}

			for (int i = 1; i <= m_iNumCells; ++i)
				m_hashtableindex[i] += m_hashtableindex[i - 1];

			std::vector<int> offsets = m_hashtableindex;
			for (int p = 0; p < m_iNumSpheres; ++p) {
				int c                          = m_hashtable[p];
				sortedParticleIds[offsets[c]++] = p;
			}
		};

		const float restDistance = 2.0f * m_particleRadius;

		for (int iter = 0; iter < numIters; ++iter) {
			std::fill(deltaVel.begin(), deltaVel.end(), glm::vec3(0.0f));
			buildSpatialHash();

			for (int p = 0; p < m_iNumSpheres; ++p) {
				glm::vec3 gp = (m_particlePos[p] + glm::vec3(0.5f)) * m_fInvSpacing;
				int ci       = std::clamp(static_cast<int>(std::floor(gp.x)), 0, m_iCellX - 1);
				int cj       = std::clamp(static_cast<int>(std::floor(gp.y)), 0, m_iCellY - 1);
				int ck       = std::clamp(static_cast<int>(std::floor(gp.z)), 0, m_iCellZ - 1);

				for (int di = -1; di <= 1; ++di) {
					int ni = ci + di;
					if (ni < 0 || ni >= m_iCellX)
						continue;
					for (int dj = -1; dj <= 1; ++dj) {
						int nj = cj + dj;
						if (nj < 0 || nj >= m_iCellY)
							continue;
						for (int dk = -1; dk <= 1; ++dk) {
							int nk = ck + dk;
							if (nk < 0 || nk >= m_iCellZ)
								continue;

							int cell  = index2GridOffset(glm::ivec3(ni, nj, nk));
							int start = m_hashtableindex[cell];
							int end   = m_hashtableindex[cell + 1];

							for (int t = start; t < end; ++t) {
								int q = sortedParticleIds[t];
								if (q <= p)
									continue;

								glm::vec3 diff = m_particlePos[p] - m_particlePos[q];
								float dist2    = glm::dot(diff, diff);
								if (dist2 >= restDistance * restDistance)
									continue;

								float dist = std::sqrt(std::max(dist2, 0.0f));
								glm::vec3 dir(1.0f, 0.0f, 0.0f);
								if (dist > kEps)
									dir = diff / dist;
								else if (((p + q) & 1) == 0)
									dir = glm::vec3(0.0f, 1.0f, 0.0f);

								float corr = 0.5f * (restDistance - dist);
								glm::vec3 delta = corr * dir;
								m_particlePos[p] += delta;
								m_particlePos[q] -= delta;
								deltaVel[p] += delta * invDt;
								deltaVel[q] -= delta * invDt;
							}
						}
					}
				}
			}

			for (int p = 0; p < m_iNumSpheres; ++p)
				m_particleVel[p] += deltaVel[p];
		}
	}

	void Simulator::handleParticleCollisions(glm::vec3 obstaclePos, float obstacleRadius, glm::vec3 obstacleVel, float timeStep) {
		(void)timeStep;
		const float minX = -0.5f + m_h + m_particleRadius;
		const float maxX = 0.5f - m_h - m_particleRadius;
		const float minY = -0.5f + m_h + m_particleRadius;
		const float maxY = 0.5f - m_h - m_particleRadius;
		const float minZ = -0.5f + m_h + m_particleRadius;
		const float maxZ = 0.5f - m_h - m_particleRadius;

		for (int p = 0; p < m_iNumSpheres; ++p) {
			glm::vec3 & pos  = m_particlePos[p];
			glm::vec3 & vel  = m_particleVel[p];

			if (pos.x < minX) {
				pos.x = minX;
				if (vel.x < 0.0f)
					vel.x = 0.0f;
			} else if (pos.x > maxX) {
				pos.x = maxX;
				if (vel.x > 0.0f)
					vel.x = 0.0f;
			}

			if (pos.y < minY) {
				pos.y = minY;
				if (vel.y < 0.0f)
					vel.y = 0.0f;
			} else if (pos.y > maxY) {
				pos.y = maxY;
				if (vel.y > 0.0f)
					vel.y = 0.0f;
			}

			if (pos.z < minZ) {
				pos.z = minZ;
				if (vel.z < 0.0f)
					vel.z = 0.0f;
			} else if (pos.z > maxZ) {
				pos.z = maxZ;
				if (vel.z > 0.0f)
					vel.z = 0.0f;
			}

			if (obstacleRadius > 0.0f) {
				glm::vec3 diff = pos - obstaclePos;
				float dist      = glm::length(diff);
				float minDist   = obstacleRadius + m_particleRadius;
				if (dist < minDist) {
					glm::vec3 n = dist > kEps ? diff / dist : glm::vec3(0.0f, 1.0f, 0.0f);
					pos          = obstaclePos + n * minDist;
					float vn     = glm::dot(vel - obstacleVel, n);
					if (vn < 0.0f)
						vel -= vn * n;
				}
			}
		}
	}

	void Simulator::updateParticleDensity() {
		std::fill(m_particleDensity.begin(), m_particleDensity.end(), 0.0f);

		const int maxI = m_iCellX - 2;
		const int maxJ = m_iCellY - 2;
		const int maxK = m_iCellZ - 2;

		for (int p = 0; p < m_iNumSpheres; ++p) {
			glm::vec3 g = (m_particlePos[p] + glm::vec3(0.5f)) * m_fInvSpacing;

			int i0 = std::clamp(static_cast<int>(std::floor(g.x)), 0, maxI);
			int j0 = std::clamp(static_cast<int>(std::floor(g.y)), 0, maxJ);
			int k0 = std::clamp(static_cast<int>(std::floor(g.z)), 0, maxK);

			float fx = clamp01(g.x - static_cast<float>(i0));
			float fy = clamp01(g.y - static_cast<float>(j0));
			float fz = clamp01(g.z - static_cast<float>(k0));

			for (int di = 0; di <= 1; ++di) {
				float wx = di ? fx : (1.0f - fx);
				for (int dj = 0; dj <= 1; ++dj) {
					float wy = dj ? fy : (1.0f - fy);
					for (int dk = 0; dk <= 1; ++dk) {
						float wz = dk ? fz : (1.0f - fz);
						int i     = i0 + di;
						int j     = j0 + dj;
						int k     = k0 + dk;
						int idx   = index2GridOffset(glm::ivec3(i, j, k));
						m_particleDensity[idx] += wx * wy * wz;
					}
				}
			}
		}

		if (m_particleRestDensity <= 0.0f) {
			float densitySum = 0.0f;
			int count        = 0;
			for (int i = 1; i < m_iCellX - 1; ++i) {
				for (int j = 1; j < m_iCellY - 1; ++j) {
					for (int k = 1; k < m_iCellZ - 1; ++k) {
						int idx = index2GridOffset(glm::ivec3(i, j, k));
						if (m_type[idx] == 1) {
							densitySum += m_particleDensity[idx];
							++count;
						}
					}
				}
			}
			if (count > 0)
				m_particleRestDensity = densitySum / static_cast<float>(count);
		}
	}

	void Simulator::transferVelocities(bool toGrid, float flipRatio) {
		const int maxI = m_iCellX - 2;
		const int maxJ = m_iCellY - 2;
		const int maxK = m_iCellZ - 2;

		auto sampleGridVelocity = [&](const glm::vec3 & pos, const std::vector<glm::vec3> & gridVel) {
			glm::vec3 g = (pos + glm::vec3(0.5f)) * m_fInvSpacing;

			int i0 = std::clamp(static_cast<int>(std::floor(g.x)), 0, maxI);
			int j0 = std::clamp(static_cast<int>(std::floor(g.y)), 0, maxJ);
			int k0 = std::clamp(static_cast<int>(std::floor(g.z)), 0, maxK);

			float fx = clamp01(g.x - static_cast<float>(i0));
			float fy = clamp01(g.y - static_cast<float>(j0));
			float fz = clamp01(g.z - static_cast<float>(k0));

			glm::vec3 out(0.0f);
			float wsum = 0.0f;
			for (int di = 0; di <= 1; ++di) {
				float wx = di ? fx : (1.0f - fx);
				for (int dj = 0; dj <= 1; ++dj) {
					float wy = dj ? fy : (1.0f - fy);
					for (int dk = 0; dk <= 1; ++dk) {
						float wz = dk ? fz : (1.0f - fz);
						float w  = wx * wy * wz;
						int idx  = index2GridOffset(glm::ivec3(i0 + di, j0 + dj, k0 + dk));
						out += w * gridVel[idx];
						wsum += w;
					}
				}
			}
			if (wsum > kEps)
				out /= wsum;
			return out;
		};

		if (toGrid) {
			m_pre_vel = m_vel;
			std::fill(m_vel.begin(), m_vel.end(), glm::vec3(0.0f));
			for (int d = 0; d < 3; ++d)
				std::fill(m_near_num[d].begin(), m_near_num[d].end(), 0.0f);

			for (int i = 0; i < m_iNumCells; ++i)
				m_type[i] = (m_s[i] == 0.0f ? 2 : 0);

			for (int p = 0; p < m_iNumSpheres; ++p) {
				glm::vec3 g = (m_particlePos[p] + glm::vec3(0.5f)) * m_fInvSpacing;
				int ci      = std::clamp(static_cast<int>(std::floor(g.x)), 0, maxI);
				int cj      = std::clamp(static_cast<int>(std::floor(g.y)), 0, maxJ);
				int ck      = std::clamp(static_cast<int>(std::floor(g.z)), 0, maxK);
				m_type[index2GridOffset(glm::ivec3(ci, cj, ck))] = 1;

				float fx = clamp01(g.x - static_cast<float>(ci));
				float fy = clamp01(g.y - static_cast<float>(cj));
				float fz = clamp01(g.z - static_cast<float>(ck));

				for (int di = 0; di <= 1; ++di) {
					float wx = di ? fx : (1.0f - fx);
					for (int dj = 0; dj <= 1; ++dj) {
						float wy = dj ? fy : (1.0f - fy);
						for (int dk = 0; dk <= 1; ++dk) {
							float wz = dk ? fz : (1.0f - fz);
							float w  = wx * wy * wz;
							int idx  = index2GridOffset(glm::ivec3(ci + di, cj + dj, ck + dk));

							m_vel[idx] += w * m_particleVel[p];
							m_near_num[0][idx] += w;
							m_near_num[1][idx] += w;
							m_near_num[2][idx] += w;
						}
					}
				}
			}

			for (int i = 0; i < m_iNumCells; ++i) {
				if (m_near_num[0][i] > kEps)
					m_vel[i].x /= m_near_num[0][i];
				if (m_near_num[1][i] > kEps)
					m_vel[i].y /= m_near_num[1][i];
				if (m_near_num[2][i] > kEps)
					m_vel[i].z /= m_near_num[2][i];
			}

			for (int i = 0; i < m_iCellX; ++i) {
				for (int j = 0; j < m_iCellY; ++j) {
					for (int k = 0; k < m_iCellZ; ++k) {
						int idx = index2GridOffset(glm::ivec3(i, j, k));
						if (m_s[idx] == 0.0f)
							m_vel[idx] = glm::vec3(0.0f);
					}
				}
			}
		} else {
			for (int p = 0; p < m_iNumSpheres; ++p) {
				glm::vec3 vPic   = sampleGridVelocity(m_particlePos[p], m_vel);
				glm::vec3 vOld   = sampleGridVelocity(m_particlePos[p], m_pre_vel);
				glm::vec3 vFlip  = m_particleVel[p] + (vPic - vOld);
				m_particleVel[p] = (1.0f - flipRatio) * vPic + flipRatio * vFlip;
			}
		}
	}

	void Simulator::solveIncompressibility(int numIters, float dt, float overRelaxation, bool compensateDrift) {
		std::fill(m_p.begin(), m_p.end(), 0.0f);

		const int n = m_iCellY * m_iCellZ;
		const int m = m_iCellZ;
		const float invDt = 1.0f / std::max(dt, 1e-6f);

		for (int iter = 0; iter < numIters; ++iter) {
			for (int i = 1; i < m_iCellX - 2; ++i) {
				for (int j = 1; j < m_iCellY - 2; ++j) {
					for (int k = 1; k < m_iCellZ - 2; ++k) {
						int idx = i * n + j * m + k;
						if (m_type[idx] != 1)
							continue;

						int idxXm = idx - n;
						int idxXp = idx + n;
						int idxYm = idx - m;
						int idxYp = idx + m;
						int idxZm = idx - 1;
						int idxZp = idx + 1;

						float sx0 = m_s[idxXm];
						float sx1 = m_s[idxXp];
						float sy0 = m_s[idxYm];
						float sy1 = m_s[idxYp];
						float sz0 = m_s[idxZm];
						float sz1 = m_s[idxZp];
						float sumS = sx0 + sx1 + sy0 + sy1 + sz0 + sz1;
						if (sumS < kEps)
							continue;

						float div = (m_vel[idxXp].x - m_vel[idx].x) +
									(m_vel[idxYp].y - m_vel[idx].y) +
									(m_vel[idxZp].z - m_vel[idx].z);

						if (compensateDrift && m_particleRestDensity > 0.0f) {
							float compression = m_particleDensity[idx] - m_particleRestDensity;
							if (compression > 0.0f)
								div -= 0.2f * compression;
						}

						float corr = -overRelaxation * div / sumS;
						m_p[idx] += corr * invDt;

						m_vel[idx].x -= sx0 * corr;
						m_vel[idxXp].x += sx1 * corr;
						m_vel[idx].y -= sy0 * corr;
						m_vel[idxYp].y += sy1 * corr;
						m_vel[idx].z -= sz0 * corr;
						m_vel[idxZp].z += sz1 * corr;
					}
				}
			}

			for (int i = 0; i < m_iCellX; ++i) {
				for (int j = 0; j < m_iCellY; ++j) {
					for (int k = 0; k < m_iCellZ; ++k) {
						int idx = index2GridOffset(glm::ivec3(i, j, k));
						if (m_s[idx] == 0.0f)
							m_vel[idx] = glm::vec3(0.0f);
					}
				}
			}
		}
	}

	void Simulator::updateParticleColors() {
		auto sampleScalar = [&](glm::vec3 const & pos, std::vector<float> const & field) {
			const int maxI = m_iCellX - 2;
			const int maxJ = m_iCellY - 2;
			const int maxK = m_iCellZ - 2;

			glm::vec3 g = (pos + glm::vec3(0.5f)) * m_fInvSpacing;
			int i0 = std::clamp(static_cast<int>(std::floor(g.x)), 0, maxI);
			int j0 = std::clamp(static_cast<int>(std::floor(g.y)), 0, maxJ);
			int k0 = std::clamp(static_cast<int>(std::floor(g.z)), 0, maxK);

			float fx = clamp01(g.x - static_cast<float>(i0));
			float fy = clamp01(g.y - static_cast<float>(j0));
			float fz = clamp01(g.z - static_cast<float>(k0));

			float value = 0.0f;
			for (int di = 0; di <= 1; ++di) {
				float wx = di ? fx : (1.0f - fx);
				for (int dj = 0; dj <= 1; ++dj) {
					float wy = dj ? fy : (1.0f - fy);
					for (int dk = 0; dk <= 1; ++dk) {
						float wz = dk ? fz : (1.0f - fz);
						int idx = index2GridOffset(glm::ivec3(i0 + di, j0 + dj, k0 + dk));
						value += wx * wy * wz * field[idx];
					}
				}
			}
			return value;
		};

		float maxAbsPressure = 1e-4f;
		for (int p = 0; p < m_iNumSpheres; ++p)
			maxAbsPressure = std::max(maxAbsPressure, std::abs(sampleScalar(m_particlePos[p], m_p)));

		for (int p = 0; p < m_iNumSpheres; ++p) {
			float speed = glm::length(m_particleVel[p]);
			float pressure = sampleScalar(m_particlePos[p], m_p);
			float density   = sampleScalar(m_particlePos[p], m_particleDensity);

			float speedFactor   = clamp01(std::pow(speed * 0.55f, 0.7f));
			float densityFactor  = m_particleRestDensity > kEps ? clamp01(density / m_particleRestDensity) : 0.0f;
			float pressureSigned = clamp01(0.5f + 0.5f * pressure / maxAbsPressure);
			float pressureFactor  = clamp01(std::abs(pressure) / maxAbsPressure);

			glm::vec3 cold(0.05f, 0.10f, 1.00f);
			glm::vec3 hot(1.00f, 0.10f, 0.05f);
			glm::vec3 color = glm::mix(cold, hot, 0.55f * speedFactor + 0.45f * pressureSigned);
			color = glm::mix(color, glm::vec3(1.0f), 0.18f * densityFactor);
			color *= 0.78f + 0.22f * pressureFactor;
			m_particleColor[p] = color;
		}
	}
} // namespace VCX::Labs::Fluid
